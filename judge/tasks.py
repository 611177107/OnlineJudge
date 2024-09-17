import dramatiq

from account.models import User
from submission.models import Submission, JudgeStatus
from judge.dispatcher import JudgeDispatcher
from utils.shortcuts import DRAMATIQ_WORKER_ARGS


@dramatiq.actor(**DRAMATIQ_WORKER_ARGS())
def judge_task(submission_id, problem_id):
    uid = Submission.objects.get(id=submission_id).user_id
    if User.objects.get(id=uid).is_disabled:
        return
    # JudgeDispatcher(submission_id, problem_id).judge()
    dispatcher = JudgeDispatcher(submission_id, problem_id)
    dispatcher.judge()

    # If not AC，then call repair function
    if Submission.objects.get(id=submission_id).result != JudgeStatus.ACCEPTED:
        dispatcher.judge_repair_code()