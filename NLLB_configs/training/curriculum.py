import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class CurriculumScheduler:

    DEFAULT_STAGES = [
        {"epoch_start": 1,  "epoch_end": 5,  "buckets": [1, 2, 3], "balanced": False},
        {"epoch_start": 6,  "epoch_end": 12, "buckets": [1, 2, 3, 4, 5], "balanced": False},
        {"epoch_start": 13, "epoch_end": 999,"buckets": [1, 2, 3, 4, 5], "balanced": True},
    ]

    def __init__(self, stages: Optional[List[dict]] = None):
        self.stages = stages or self.DEFAULT_STAGES

    def get_stage(self, epoch: int) -> dict:
        for stage in self.stages:
            if stage["epoch_start"] <= epoch <= stage["epoch_end"]:
                return stage
        return self.stages[-1]

    def get_active_buckets(self, epoch: int) -> List[int]:
        return self.get_stage(epoch)["buckets"]

    def use_balanced_sampling(self, epoch: int) -> bool:
        return self.get_stage(epoch).get("balanced", False)

    def log_stage(self, epoch: int):
        stage = self.get_stage(epoch)
        logger.info(
            f"[Curriculum] Epoch {epoch}: "
            f"active_buckets={stage['buckets']}, "
            f"balanced_sampling={stage.get('balanced', False)}"
        )
