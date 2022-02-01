"""
Extensions to plug into Plumber's pipeline optimizer to implement new behavior.
"""

import logging

import graphsurgeon
import tensorflow as tf

from plumber_analysis import graphdef_util, gen_util, pipeline_optimizer

def instantiate_pipeline(graphdef, surgeon=None):
    if surgeon is None:
        surgeon = graphsurgeon.StaticGraph(graphdef)
    element_spec = graphdef_util.element_spec_from_graph(surgeon)
    ds = graphdef_util.instantiate_pipeline(graphdef, element_spec)
    return ds

def graphdef_to_plumber(graphdef, benchmark_interval_s=22, stats_filename=None):
    dataset = instantiate_pipeline(graphdef)
    options = tf.data.Options()
    if stats_filename is None:
        stats_filename = "_tmp_stats.pb"
    gen_util.add_analysis_to_dataset_options(
        options, hard_fail=True, stats_filename=stats_filename,
        dump_period_s=5)
    dataset = dataset.with_options(options)
    _ = gen_util.benchmark_dataset(dataset, time_limit_s=benchmark_interval_s)
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(stats_filename)
    return plumber

class GraphObject:
    """Keeps track of at least the plumber object, and potentially any intermediate changes.
    """
    def __init__(self, plumber, graphdef=None):
        self.plumber = plumber
        self._model = None
        if graphdef is None:
            self._graphdef = self.model().graphdef()
        else:
            self._graphdef = graphdef

    def model(self):
        if not self._model:
            self._model = self.plumber.model()
        return self._model

    def graphdef(self):
        return self._graphdef

    def surgeon(self, dynamic=False):
        if dynamic:
            return graphsurgeon.DynamicGraph(self.graphdef())
        else:
            return graphsurgeon.StaticGraph(self.graphdef())

    def instantiate(self):
        graphdef = self.graphdef()
        surgeon = graphsurgeon.StaticGraph(graphdef)
        return instantiate_pipeline(graphdef, surgeon)

    def commit_modification(self, graphdef, plumber=None, update_plumber=True):
        if plumber is None:
            if update_plumber:
                # NOTE(mkuchnik): This can be VERY dangerous to keep in sync
                logging.info("Extension materializing new plumber")
                plumber = graphdef_to_plumber(graphdef)
                self.plumber = plumber
            return self
        else:
            raise NotImplementedError("Modifying plumber not supported")

class Extension(object):
    """A Plumber extension.

    An extension is a transaction applied to the graph_object.
    If the checks fail, it is not applied.
    TODO(mkuchnik): Take machine info"""

    def apply_transaction_if_possible(self, graph_object: GraphObject):
        is_same = True
        if self._precondition(graph_object):
            new_graph_object, is_same = self._apply(graph_object)
            if self._postcondition(new_graph_object):
                graph_object = new_graph_object
        return graph_object, is_same

    def _precondition(self, graph_object: GraphObject) -> bool:
        pass

    def _postcondition(self, graph_object: GraphObject) -> bool:
        pass

    def _apply(self, graph_object: GraphObject):
        """Object and if same  -> GraphObject, bool"""
        pass


class RemoveCaches(Extension):
    """Remove all caches from the graph_object e.g., to prevent OOM"""

    def _precondition(self, graph_object: GraphObject) -> bool:
        return True

    def _postcondition(self, graph_object: GraphObject) -> bool:
        return True

    def _apply(self, graph_object: GraphObject) -> GraphObject:
        graphdef = graph_object.graphdef()
        new_graphdef, num_removed = graphdef_util.remove_caching_datasets(
                graphdef, return_num_removed=True)
        logging.info("RemoveCache removed {} nodes".format(num_removed))
        # NOTE(mkuchnik): compat check has false positives
        is_same, diff = pipeline_optimizer.check_graphdef_compatibility(
                new_graphdef, graphdef)
        no_caches_removed = bool(num_removed == 0)
        if is_same != no_caches_removed:
            logging.warning(
                    "Expected same: {}, non removed: {}, but found: {} and {} removed".format(
                        is_same, no_caches_removed, diff, num_removed))
        if not is_same:
            logging.info("Remove Caches Detected a Change: {}".format(diff))
        new_graph_object = graph_object.commit_modification(
                new_graphdef, update_plumber=not no_caches_removed)
        return new_graph_object, no_caches_removed
