from overrides import overrides

from allennlp.training.metrics.metric import Metric


@Metric.register("acc_vs_del")
class AccuracyVSDeletion(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self, del_threshold=0.1, aggr="diff") -> None:
        self._total_acc = 0.0
        self._total_del = 0.0
        self._count = 0
        self._del_threshold = del_threshold
        self._aggr = aggr

    @overrides
    def __call__(self, acc, delete):
        """
        Parameters
        ----------
        """
        self._total_acc += acc
        self._total_del += delete
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_acc = self._total_acc / self._count if self._count > 0 else 0
        average_del = self._total_del / self._count if self._count > 0 else 0
        average_del = max(average_del, self._del_threshold)
        if reset:
            self.reset()
        if self._aggr == "diff":
            return average_acc - average_del
        elif self._aggr == "sum":
            return average_acc + average_del
        else:
            raise Exception("{} not implemented".format(self._aggr))

    @overrides
    def reset(self):
        self._total_acc = 0.0
        self._total_del = 0.0
        self._count = 0
