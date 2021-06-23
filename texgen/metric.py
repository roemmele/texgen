from typing import Any, Dict

class RawMetric():

    metric_name = None

    @classmethod
    def get_metric_names(cls) -> Dict[Any, Any]:
        names = {metric_cls.get_name(): metric_cls
                 for metric_cls in cls.__subclasses__()}
        return names

    @classmethod
    def get_metric_from_name(cls, name):
        name = name.strip().lower()
        all_names = cls.get_metric_names()
        assert name in all_names, "metric name {} not found. existing metrics are: {}".format(
            name, list(all_names.keys()))
        return all_names[name]

    @classmethod
    def get_name(cls):
        return cls.metric_name
