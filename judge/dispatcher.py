import hashlib
import json
import logging
from urllib.parse import urljoin

import requests
from django.db import transaction, IntegrityError
from django.db.models import F

from account.models import User
from conf.models import JudgeServer
from contest.models import ContestRuleType, ACMContestRank, OIContestRank, ContestStatus
from options.options import SysOptions
from problem.models import Problem, ProblemRuleType
from problem.utils import parse_problem_template
from submission.models import JudgeStatus, Submission
from utils.cache import cache
from utils.constants import CacheKey

from openai import OpenAI
from django.conf import settings
from repair_result.models import RepairResult

logger = logging.getLogger(__name__)


# 继续处理在队列中的问题
def process_pending_task():
    if cache.llen(CacheKey.waiting_queue):
        # 防止循环引入
        from judge.tasks import judge_task
        tmp_data = cache.rpop(CacheKey.waiting_queue)
        if tmp_data:
            data = json.loads(tmp_data.decode("utf-8"))
            judge_task.send(**data)


class ChooseJudgeServer:
    def __init__(self):
        self.server = None

    def __enter__(self) -> [JudgeServer, None]:
        with transaction.atomic():
            servers = JudgeServer.objects.select_for_update().filter(is_disabled=False).order_by("task_number")
            servers = [s for s in servers if s.status == "normal"]
            for server in servers:
                if server.task_number <= server.cpu_core * 2:
                    server.task_number = F("task_number") + 1
                    server.save(update_fields=["task_number"])
                    self.server = server
                    return server
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.server:
            JudgeServer.objects.filter(id=self.server.id).update(task_number=F("task_number") - 1)


class DispatcherBase(object):
    def __init__(self):
        self.token = hashlib.sha256(SysOptions.judge_server_token.encode("utf-8")).hexdigest()

    def _request(self, url, data=None):
        kwargs = {"headers": {"X-Judge-Server-Token": self.token}}
        if data:
            kwargs["json"] = data
        try:
            return requests.post(url, **kwargs).json()
        except Exception as e:
            logger.exception(e)


class SPJCompiler(DispatcherBase):
    def __init__(self, spj_code, spj_version, spj_language):
        super().__init__()
        spj_compile_config = list(filter(lambda config: spj_language == config["name"], SysOptions.spj_languages))[0]["spj"][
            "compile"]
        self.data = {
            "src": spj_code,
            "spj_version": spj_version,
            "spj_compile_config": spj_compile_config
        }

    def compile_spj(self):
        with ChooseJudgeServer() as server:
            if not server:
                return "No available judge_server"
            result = self._request(urljoin(server.service_url, "compile_spj"), data=self.data)
            if not result:
                return "Failed to call judge server"
            if result["err"]:
                return result["data"]


class JudgeDispatcher(DispatcherBase):
    def __init__(self, submission_id, problem_id):
        super().__init__()
        self.submission = Submission.objects.get(id=submission_id)
        self.contest_id = self.submission.contest_id
        self.last_result = self.submission.result if self.submission.info else None

        if self.contest_id:
            self.problem = Problem.objects.select_related("contest").get(id=problem_id, contest_id=self.contest_id)
            self.contest = self.problem.contest
        else:
            self.problem = Problem.objects.get(id=problem_id)

    def _compute_statistic_info(self, resp_data):
        # 用时和内存占用保存为多个测试点中最长的那个
        self.submission.statistic_info["time_cost"] = max([x["cpu_time"] for x in resp_data])
        self.submission.statistic_info["memory_cost"] = max([x["memory"] for x in resp_data])

        # sum up the score in OI mode
        if self.problem.rule_type == ProblemRuleType.OI:
            score = 0
            try:
                for i in range(len(resp_data)):
                    if resp_data[i]["result"] == JudgeStatus.ACCEPTED:
                        resp_data[i]["score"] = self.problem.test_case_score[i]["score"]
                        score += resp_data[i]["score"]
                    else:
                        resp_data[i]["score"] = 0
            except IndexError:
                logger.error(f"Index Error raised when summing up the score in problem {self.problem.id}")
                self.submission.statistic_info["score"] = 0
                return
            self.submission.statistic_info["score"] = score

    def judge(self):
        print("------print message: start judge the code")
        logger.info("------logger message: start judge the code")
        language = self.submission.language
        sub_config = list(filter(lambda item: language == item["name"], SysOptions.languages))[0]
        spj_config = {}
        if self.problem.spj_code:
            for lang in SysOptions.spj_languages:
                if lang["name"] == self.problem.spj_language:
                    spj_config = lang["spj"]
                    break

        if language in self.problem.template:
            template = parse_problem_template(self.problem.template[language])
            code = f"{template['prepend']}\n{self.submission.code}\n{template['append']}"
        else:
            code = self.submission.code

        data = {
            "language_config": sub_config["config"],
            "src": code,
            "max_cpu_time": self.problem.time_limit,
            "max_memory": 1024 * 1024 * self.problem.memory_limit,
            "test_case_id": self.problem.test_case_id,
            "output": False,
            "spj_version": self.problem.spj_version,
            "spj_config": spj_config.get("config"),
            "spj_compile_config": spj_config.get("compile"),
            "spj_src": self.problem.spj_code,
            "io_mode": self.problem.io_mode
        }

        with ChooseJudgeServer() as server:
            if not server:
                data = {"submission_id": self.submission.id, "problem_id": self.problem.id}
                cache.lpush(CacheKey.waiting_queue, json.dumps(data))
                return
            Submission.objects.filter(id=self.submission.id).update(result=JudgeStatus.JUDGING)
            resp = self._request(urljoin(server.service_url, "/judge"), data=data)

        if not resp:
            Submission.objects.filter(id=self.submission.id).update(result=JudgeStatus.SYSTEM_ERROR)
            return

        if resp["err"]:
            self.submission.result = JudgeStatus.COMPILE_ERROR
            self.submission.statistic_info["err_info"] = resp["data"]
            self.submission.statistic_info["score"] = 0
        else:
            resp["data"].sort(key=lambda x: int(x["test_case"]))
            self.submission.info = resp
            self._compute_statistic_info(resp["data"])
            error_test_case = list(filter(lambda case: case["result"] != 0, resp["data"]))
            # ACM模式下,多个测试点全部正确则AC，否则取第一个错误的测试点的状态
            # OI模式下, 若多个测试点全部正确则AC， 若全部错误则取第一个错误测试点状态，否则为部分正确
            if not error_test_case:
                self.submission.result = JudgeStatus.ACCEPTED
            elif self.problem.rule_type == ProblemRuleType.ACM or len(error_test_case) == len(resp["data"]):
                self.submission.result = error_test_case[0]["result"]
                gpt_message_result = self.sendMessageToGPT(code)
                # gpt_message_rusult = "test gpt message row2"
                # Submission.objects.filter(id=self.submission.id).update(gpt_message=gpt_message_rusult)
                self.submission.gpt_message = gpt_message_result
            else:
                self.submission.result = JudgeStatus.PARTIALLY_ACCEPTED
                gpt_message_result = self.sendMessageToGPT(code)
                # gpt_message_result = "test gpt message row2"
                # Submission.objects.filter(id=self.submission.id).update(gpt_message=gpt_message_rusult)
                self.submission.gpt_message = gpt_message_result
        self.submission.save()

        if self.contest_id:
            if self.contest.status != ContestStatus.CONTEST_UNDERWAY or \
                    User.objects.get(id=self.submission.user_id).is_contest_admin(self.contest):
                logger.info(
                    "Contest debug mode, id: " + str(self.contest_id) + ", submission id: " + self.submission.id)
                return
            with transaction.atomic():
                self.update_contest_problem_status()
                self.update_contest_rank()
        else:
            if self.last_result:
                self.update_problem_status_rejudge()
            else:
                self.update_problem_status()

        # 至此判题结束，尝试处理任务队列中剩余的任务
        process_pending_task()

    def judge_repair_code(self):
        repair_models = {
            'model1': self.sendCodeToModel1,
            'model2': self.sendCodeToModel2,
            # 'chain0': self.sendCodeToChain0,
            # 'chain10': self.sendCodeToChain10,
            'chain1': self.sendCodeToChain1,
            'chain2': self.sendCodeToChain2,
            'chain3': self.sendCodeToChain3,
            # 'chain11': self.sendCodeToChain11,
            'chain4': self.sendCodeToChain4,
            # 'chain5': self.sendCodeToChain5,
            # 'chain6': self.sendCodeToChain6,
            # 'chain12': self.sendCodeToChain12,
            # 'chain7': self.sendCodeToChain7,
            # 'chain8': self.sendCodeToChain8,
            # 'chain9': self.sendCodeToChain9,
            # 'mychain1':self.sendCodeToMyChain1,

            # 'test_model': self.testModelMessage,
        }
        # RepairResult.objects.update_or_create(
        #             defaults={
        #                 'repair_code': "test repair_code",
        #                 'repair_result': 1
        #             },
        #             submission=self.submission,
        #             model_name="test model_name"
        #         )
        code = self.submission.code
        # test_cases = self.problem.test_cases

        for model_name, model_api_func in repair_models.items():
            try:

                repair_code = model_api_func(code)    
                result = self._judge_code(repair_code)

                print(f"------print message: result of the repair code is {result}")
                logger.info(f"------logger message: result of the repair code is {result}")
                RepairResult.objects.update_or_create(
                    defaults={
                        'repair_code': repair_code,
                        'repair_result': result
                    },
                    submission=self.submission,
                    model_name=model_name
                )
            except Exception as e:
                logger.error(f"Failed to call {model_name} API: {e}")
                RepairResult.objects.update_or_create(
                    defaults={
                        'repair_code': f"Failed to call {model_name} API: {e}",
                        'repair_result': 1
                    },
                    submission=self.submission,
                    model_name="test model_name error"
                )

    def _judge_code(self, code):
        language = self.submission.language
        sub_config = list(filter(lambda item: language == item["name"], SysOptions.languages))[0]
        spj_config = {}
        if self.problem.spj_code:
            for lang in SysOptions.spj_languages:
                if lang["name"] == self.problem.spj_language:
                    spj_config = lang["spj"]
                    break

        if language in self.problem.template:
            template = parse_problem_template(self.problem.template[language])
            code = f"{template['prepend']}\n{code}\n{template['append']}"
        else:
            code = code

        data = {
            "language_config": sub_config["config"],
            "src": code,
            "max_cpu_time": self.problem.time_limit,
            "max_memory": 1024 * 1024 * self.problem.memory_limit,
            "test_case_id": self.problem.test_case_id,
            "output": False,
            "spj_version": self.problem.spj_version,
            "spj_config": spj_config.get("config"),
            "spj_compile_config": spj_config.get("compile"),
            "spj_src": self.problem.spj_code,
            "io_mode": self.problem.io_mode
        }

        with ChooseJudgeServer() as server:
            if not server:
                # return {"result": JudgeStatus.SYSTEM_ERROR, "statistic_info": {}}
                return JudgeStatus.SYSTEM_ERROR
            resp = self._request(urljoin(server.service_url, "/judge"), data=data)

        if not resp:
            # return {"result": JudgeStatus.SYSTEM_ERROR, "statistic_info": {}}
            return JudgeStatus.SYSTEM_ERROR

        if resp["err"]:
            # return {"result": JudgeStatus.COMPILE_ERROR, "statistic_info": {"err_info": resp["data"], "score": 0}}
            return JudgeStatus.COMPILE_ERROR
        else:
            resp["data"].sort(key=lambda x: int(x["test_case"]))
            # statistic_info = self._compute_statistic_info(resp["data"])
            error_test_case = list(filter(lambda case: case["result"] != 0, resp["data"]))

            result = JudgeStatus.ACCEPTED if not error_test_case else (
                JudgeStatus.PARTIALLY_ACCEPTED if self.problem.rule_type == ProblemRuleType.OI else error_test_case[0]["result"]
            )

            # return {"result": result, "statistic_info": statistic_info}
            return result

    def update_problem_status_rejudge(self):
        result = str(self.submission.result)
        problem_id = str(self.problem.id)
        with transaction.atomic():
            # update problem status
            problem = Problem.objects.select_for_update().get(contest_id=self.contest_id, id=self.problem.id)
            if self.last_result != JudgeStatus.ACCEPTED and self.submission.result == JudgeStatus.ACCEPTED:
                problem.accepted_number += 1
            problem_info = problem.statistic_info
            problem_info[self.last_result] = problem_info.get(self.last_result, 1) - 1
            problem_info[result] = problem_info.get(result, 0) + 1
            problem.save(update_fields=["accepted_number", "statistic_info"])

            profile = User.objects.select_for_update().get(id=self.submission.user_id).userprofile
            if problem.rule_type == ProblemRuleType.ACM:
                acm_problems_status = profile.acm_problems_status.get("problems", {})
                if acm_problems_status[problem_id]["status"] != JudgeStatus.ACCEPTED:
                    acm_problems_status[problem_id]["status"] = self.submission.result
                    if self.submission.result == JudgeStatus.ACCEPTED:
                        profile.accepted_number += 1
                profile.acm_problems_status["problems"] = acm_problems_status
                profile.save(update_fields=["accepted_number", "acm_problems_status"])

            else:
                oi_problems_status = profile.oi_problems_status.get("problems", {})
                score = self.submission.statistic_info["score"]
                if oi_problems_status[problem_id]["status"] != JudgeStatus.ACCEPTED:
                    # minus last time score, add this tim score
                    profile.add_score(this_time_score=score,
                                      last_time_score=oi_problems_status[problem_id]["score"])
                    oi_problems_status[problem_id]["score"] = score
                    oi_problems_status[problem_id]["status"] = self.submission.result
                    if self.submission.result == JudgeStatus.ACCEPTED:
                        profile.accepted_number += 1
                profile.oi_problems_status["problems"] = oi_problems_status
                profile.save(update_fields=["accepted_number", "oi_problems_status"])

    def update_problem_status(self):
        result = str(self.submission.result)
        problem_id = str(self.problem.id)
        with transaction.atomic():
            # update problem status
            problem = Problem.objects.select_for_update().get(contest_id=self.contest_id, id=self.problem.id)
            problem.submission_number += 1
            if self.submission.result == JudgeStatus.ACCEPTED:
                problem.accepted_number += 1
            problem_info = problem.statistic_info
            problem_info[result] = problem_info.get(result, 0) + 1
            problem.save(update_fields=["accepted_number", "submission_number", "statistic_info"])

            # update_userprofile
            user = User.objects.select_for_update().get(id=self.submission.user_id)
            user_profile = user.userprofile
            user_profile.submission_number += 1
            if problem.rule_type == ProblemRuleType.ACM:
                acm_problems_status = user_profile.acm_problems_status.get("problems", {})
                if problem_id not in acm_problems_status:
                    acm_problems_status[problem_id] = {"status": self.submission.result, "_id": self.problem._id}
                    if self.submission.result == JudgeStatus.ACCEPTED:
                        user_profile.accepted_number += 1
                elif acm_problems_status[problem_id]["status"] != JudgeStatus.ACCEPTED:
                    acm_problems_status[problem_id]["status"] = self.submission.result
                    if self.submission.result == JudgeStatus.ACCEPTED:
                        user_profile.accepted_number += 1
                user_profile.acm_problems_status["problems"] = acm_problems_status
                user_profile.save(update_fields=["submission_number", "accepted_number", "acm_problems_status"])

            else:
                oi_problems_status = user_profile.oi_problems_status.get("problems", {})
                score = self.submission.statistic_info["score"]
                if problem_id not in oi_problems_status:
                    user_profile.add_score(score)
                    oi_problems_status[problem_id] = {"status": self.submission.result,
                                                      "_id": self.problem._id,
                                                      "score": score}
                    if self.submission.result == JudgeStatus.ACCEPTED:
                        user_profile.accepted_number += 1
                elif oi_problems_status[problem_id]["status"] != JudgeStatus.ACCEPTED:
                    # minus last time score, add this time score
                    user_profile.add_score(this_time_score=score,
                                           last_time_score=oi_problems_status[problem_id]["score"])
                    oi_problems_status[problem_id]["score"] = score
                    oi_problems_status[problem_id]["status"] = self.submission.result
                    if self.submission.result == JudgeStatus.ACCEPTED:
                        user_profile.accepted_number += 1
                user_profile.oi_problems_status["problems"] = oi_problems_status
                user_profile.save(update_fields=["submission_number", "accepted_number", "oi_problems_status"])

    def update_contest_problem_status(self):
        with transaction.atomic():
            user = User.objects.select_for_update().get(id=self.submission.user_id)
            user_profile = user.userprofile
            problem_id = str(self.problem.id)
            if self.contest.rule_type == ContestRuleType.ACM:
                contest_problems_status = user_profile.acm_problems_status.get("contest_problems", {})
                if problem_id not in contest_problems_status:
                    contest_problems_status[problem_id] = {"status": self.submission.result, "_id": self.problem._id}
                elif contest_problems_status[problem_id]["status"] != JudgeStatus.ACCEPTED:
                    contest_problems_status[problem_id]["status"] = self.submission.result
                else:
                    # 如果已AC， 直接跳过 不计入任何计数器
                    return
                user_profile.acm_problems_status["contest_problems"] = contest_problems_status
                user_profile.save(update_fields=["acm_problems_status"])

            elif self.contest.rule_type == ContestRuleType.OI:
                contest_problems_status = user_profile.oi_problems_status.get("contest_problems", {})
                score = self.submission.statistic_info["score"]
                if problem_id not in contest_problems_status:
                    contest_problems_status[problem_id] = {"status": self.submission.result,
                                                           "_id": self.problem._id,
                                                           "score": score}
                else:
                    contest_problems_status[problem_id]["score"] = score
                    contest_problems_status[problem_id]["status"] = self.submission.result
                user_profile.oi_problems_status["contest_problems"] = contest_problems_status
                user_profile.save(update_fields=["oi_problems_status"])

            problem = Problem.objects.select_for_update().get(contest_id=self.contest_id, id=self.problem.id)
            result = str(self.submission.result)
            problem_info = problem.statistic_info
            problem_info[result] = problem_info.get(result, 0) + 1
            problem.submission_number += 1
            if self.submission.result == JudgeStatus.ACCEPTED:
                problem.accepted_number += 1
            problem.save(update_fields=["submission_number", "accepted_number", "statistic_info"])

    def update_contest_rank(self):
        if self.contest.rule_type == ContestRuleType.OI or self.contest.real_time_rank:
            cache.delete(f"{CacheKey.contest_rank_cache}:{self.contest.id}")

        def get_rank(model):
            return model.objects.select_for_update().get(user_id=self.submission.user_id, contest=self.contest)

        if self.contest.rule_type == ContestRuleType.ACM:
            model = ACMContestRank
            func = self._update_acm_contest_rank
        else:
            model = OIContestRank
            func = self._update_oi_contest_rank

        try:
            rank = get_rank(model)
        except model.DoesNotExist:
            try:
                model.objects.create(user_id=self.submission.user_id, contest=self.contest)
                rank = get_rank(model)
            except IntegrityError:
                rank = get_rank(model)
        func(rank)

    def _update_acm_contest_rank(self, rank):
        info = rank.submission_info.get(str(self.submission.problem_id))
        # 因前面更改过，这里需要重新获取
        problem = Problem.objects.select_for_update().get(contest_id=self.contest_id, id=self.problem.id)
        # 此题提交过
        if info:
            if info["is_ac"]:
                return

            rank.submission_number += 1
            if self.submission.result == JudgeStatus.ACCEPTED:
                rank.accepted_number += 1
                info["is_ac"] = True
                info["ac_time"] = (self.submission.create_time - self.contest.start_time).total_seconds()
                rank.total_time += info["ac_time"] + info["error_number"] * 20 * 60

                if problem.accepted_number == 1:
                    info["is_first_ac"] = True
            elif self.submission.result != JudgeStatus.COMPILE_ERROR:
                info["error_number"] += 1

        # 第一次提交
        else:
            rank.submission_number += 1
            info = {"is_ac": False, "ac_time": 0, "error_number": 0, "is_first_ac": False}
            if self.submission.result == JudgeStatus.ACCEPTED:
                rank.accepted_number += 1
                info["is_ac"] = True
                info["ac_time"] = (self.submission.create_time - self.contest.start_time).total_seconds()
                rank.total_time += info["ac_time"]

                if problem.accepted_number == 1:
                    info["is_first_ac"] = True

            elif self.submission.result != JudgeStatus.COMPILE_ERROR:
                info["error_number"] = 1
        rank.submission_info[str(self.submission.problem_id)] = info
        rank.save()

    def _update_oi_contest_rank(self, rank):
        problem_id = str(self.submission.problem_id)
        current_score = self.submission.statistic_info["score"]
        last_score = rank.submission_info.get(problem_id)
        if last_score:
            rank.total_score = rank.total_score - last_score + current_score
        else:
            rank.total_score = rank.total_score + current_score
        rank.submission_info[problem_id] = current_score
        rank.save()

    def testModelMessage(self, code):
        code = f"this is code:{code}"
        return "return message from testModelMessage"

    def sendCodeToModel1(self, code):
        api_url = "http://140.127.74.147:8801/model1/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })
        
        return response.json().get('output')

    def sendCodeToModel2(self, code):
        api_url = "http://140.127.74.147:8801/model2/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })
        
        return response.json().get('output')

    def sendCodeToChain0(self, code):
        api_url = "http://140.127.74.147:8801/chain0/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')

    def sendCodeToChain1(self, code):
        api_url = "http://140.127.74.147:8801/chain1/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')
    
    def sendCodeToChain2(self, code):
        api_url = "http://140.127.74.147:8801/chain2/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')
    
    def sendCodeToChain3(self, code):
        api_url = "http://140.127.74.147:8801/chain3/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')
    
    def sendCodeToChain4(self, code):
        api_url = "http://140.127.74.147:8801/chain4/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')
    
    def sendCodeToChain5(self, code):
        api_url = "http://140.127.74.147:8801/chain5/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')
    
    def sendCodeToChain6(self, code):
        api_url = "http://140.127.74.147:8801/chain6/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')

    def sendCodeToChain7(self, code):
        api_url = "http://140.127.74.147:8801/chain7/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')

    def sendCodeToChain8(self, code):
        api_url = "http://140.127.74.147:8801/chain8/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')

    def sendCodeToChain9(self, code):
        api_url = "http://140.127.74.147:8801/chain9/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')

    def sendCodeToChain10(self, code):
        api_url = "http://140.127.74.147:8801/chain10/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')

    def sendCodeToChain11(self, code):
        api_url = "http://140.127.74.147:8801/chain11/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')

    def sendCodeToChain12(self, code):
        api_url = "http://140.127.74.147:8801/chain12/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')
    

    def sendCodeToMyChain1(self, code):
        api_url = "http://140.127.74.147:8801/mychain1/invoke"
        response = requests.post(api_url, json={
            "input": {
                "code": code
            }
        })

        return response.json().get('output')


    def sendMessageToGPT(self, code):
        try:
            client = OpenAI(
                # This is the default and can be omitted
                # My OpenAI API key
                api_key=settings.OPENAI_API_KEY
            )

            # chat_completion = client.chat.completions.create(
            #     messages=[
            #     {
            #         "role": "system",
            #         "content": "以下程式碼含有錯誤，請幫我修復他",
            #     },
            #     {
            #         "role": "user",
            #         "content": code,
            #     }
            #     ],
            #     model="gpt-3.5-turbo",
            # )


            chat_completion = client.chat.completions.create(
                messages=[
                {
                        "role": "system",
                    "content": "您是一位有用的程式碼修復助手，可以替工程師錯誤的程式提供修復協助。不提供修正的程式碼，只對程式碼錯誤的部分進行解釋，並且只使用台灣用語的繁體中文",
                },
                {"role": "user",
                 "content": """def counting_sort(A):
  max_element = max(A)
  count = [0] * (max_element + 1)

  for num in A:
      count[num] += 1

  for i in range(1, len(count)):
      count[i] += count[i - 1]

  output = [0] * len(A)

  
  for num in reversed(A):
      output[count[num] - 1] = num
      count[num] -= 1

  return output


if __name__== "__main__":

  n = int(input("")) # 輸入數字的個數
  A = list(map(int, input("").split())) # 輸入數列，並轉換為整數列表

  A = counting_sort(A)

  for j in A:
    print(j, end=" ")
  print()"""},
        {"role": "assistant", 
         "content": """這段程式碼是計數排序 (Counting Sort) 的實作，用來對一個數列進行排序。計數排序的基本原理是統計每個元素出現的次數，然後根據這個統計信息來將數列排序。程式碼本身的邏輯是正確的，但有一個較明顯的錯誤：

1. 在執行計數排序前，沒有考慮到數列中可能存在負數的情況。因此在確定計數陣列大小時，直接以最大元素值為基準可能會導致陣列溢位。

為了修正這個問題，建議您在計算最大元素值時，考慮到數列中可能存在負數的情況。您可以透過計算最小值和最大值來確定計數陣列的大小，修正方式如下：

```python
min_element = min(A)
max_element = max(A)
count = [0] * (max_element - min_element + 1)

# 將每個元素映射到計數陣列的索引位置，需要調整索引的計算方式
count[num - min_element] += 1
```

在修正完這部分後，您的計數排序程式碼就可以正確地處理包含負數的數列了。
"""},
                {"role": "user", 
                 "content": """def quicksort(arr):
    n = len(arr)
    if n <= 1:
        return arr
    leftArr = []
    rightArr = []
    pivot = arr[0]
    for i in range(1,n):
        if arr[i] < pivot:
            leftArr.append(arr[i])
        else:
            rightArr.append(arr[i])
    return quicksort(leftArr)+[pivot]+quicksort(rightArr)

while True:
    try:
        n = int(input())
        if n==0 or n>1000:
            break
        data = input().split(" ")
        quicksort(data)
        for num in quicksort(data):
            print(num, end=" ")
    except:
        break"""},
        {"role": "assistant", 
         "content": """這段程式碼包含了快速排序演算法的實現以及對數組的排序。但是在程式碼中存在著幾個錯誤：

1. 函數 `quicksort` 在每次遞迴調用後都會返回排序後的結果，但是在 `while True` 迴圈中，並未將這個排序後的結果賦值給任何變量，所以即使排序操作執行完畢，結果未被使用。

2. 快速排序演算法處理的是數字，但程式中 `input().split(" ")` 返回的是字符串而非整數，在進行比較時將會出錯。

針對第一個問題，您可以將排序後的結果存放在一個變數中，再統一輸出結果，例如：

```python
sorted_data = quicksort(data)
for num in sorted_data:
    print(num, end=" ")
```

針對第二個問題，您需要將輸入的字符串轉換為整數，可以使用 `map` 函數將其應用於所有內容，如下所示：

```python
data = list(map(int, input().split(" ")))
```

以上修正後的程式碼應該可以正確地執行並輸出排序後的結果。
"""},
                {"role": "user", 
                 "content": """def activity_selection(start, end):
    n = len(start)
    activities = []  # Selected activities
    activities.append(0)  # Always select the first activity
    prev_end = end[0]  # End time of the first activity

    # Iterate over the remaining activities
    for i in range(1, n):
        if start[i] >= prev_end:
            activities.append(i)
            prev_end = end[i]

    return activities

# Input
start_time = list(map(int, input().split()))
end_time = list(map(int, input().split()))

# Call the activity selection function
selected_activities = activity_selection(start_time, end_time)

# Output
print(len(selected_activities))
for activity in selected_activities:
    print(activity, end=" ")
print()"""},
        {"role": "assistant", 
         "content": """這段程式碼是一個活動選擇問題的求解程式，主要根據活動的開始時間和結束時間來找出最大可能的活動組合。

然而，程式碼中存在一個潛在的問題：

1. 在呼叫 `activity_selection` 函數時，用來儲存活動選擇結果的 `selected_activities` 變數實際上儲存的是被選擇的活動的索引位置，而不是活動序號。這點在印出結果時可能會造成混淆。建議修改一下顯示的方式，讓輸出更易讀。

如果您想要讓輸出顯示活動的序號而非索引位置，您可以在印出前對索引值加一，將其轉換為活動的序號。例如：

```python
print(len(selected_activities))
for index in selected_activities:
    print(index + 1, end=" ")
print()
```

這樣修改後的程式碼將會更清楚地顯示被選擇的活動序號。
"""},
                {
                    "role": "user",
                    "content": code,
                }
                ],
                model="gpt-3.5-turbo",
            )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            return str(e)