from criteria.core.tasks.accuracy_task import AccuracyTask
from criteria.core.tasks.clicking_task import ClickingTask
from criteria.core.tasks.movement_task import MovementTask
from criteria.core.tasks.tracking_task import TrackingTask

TASK_SEQUENCE = [MovementTask, AccuracyTask, TrackingTask, ClickingTask]
TASK_IDS = [task.id for task in TASK_SEQUENCE]

