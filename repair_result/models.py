from django.db import models
from submission.models import Submission
from submission.models import JudgeStatus


class RepairResult(models.Model):
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE, related_name='repair_results')
    model_name = models.TextField()
    repair_code = models.TextField()
    # repair_result = JSONField()
    repair_result = models.IntegerField(db_index=True, default=JudgeStatus.PENDING)
    repair_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'repair_result'
        ordering = ('-repair_time',)  # 根據修復時間進行排序
        indexes = [
            models.Index(fields=['submission']),
            models.Index(fields=['model_name']),
        ]

    def __str__(self):
        return f"{self.model_name} - {self.submission.id}"
