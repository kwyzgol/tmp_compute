from dataclasses import dataclass


class GTime:
    @classmethod
    def milliseconds(cls, time: float):
        return time*0.001

    @classmethod
    def seconds(cls, time: float):
        return time

    @classmethod
    def minutes(cls, time: float):
        return time*60


@dataclass
class TimeManager:
    interval: float = GTime.milliseconds(250)
    episode_time_limit: float = GTime.minutes(10)

    def get_episode_max_steps(self):
        return int(self.episode_time_limit / self.interval)

    def duration(self, time: float):
        return time / self.interval

    def value_per_second(self, value: float):
        return value * self.interval

    def fps(self):
        return int(1 / self.interval)