import tvm

from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.DietCodeDispatcher")
class DietCodeDispatcher(Object):

    def dispatch(self, shape_tuple):
        sched, in_args = _ffi_api.DispatcherDispatchAndApplySteps(
                             self, self._find_wkl_id(shape_tuple)
                         )
        return sched, in_args

    def dispatch_to_state(self, shape_tuple):
        return _ffi_api.DispatcherDispatch(
                   self, self._find_wkl_id(shape_tuple)
               )

    @property
    def states(self):
        return _ffi_api.DispatcherStates(self)

    @property
    def inst_disp_map(self):
        return _ffi_api.DispatcherInstDispMap(self)

    def _find_wkl_id(self, shape_tuple):
        from tvm.ir import Array

        if isinstance(shape_tuple, Array):
            shape_tuple = tuple([int(v) for v in list(shape_tuple)])
        for i, wkl_inst in enumerate(list(self.search_task.wkl_insts)):
            wkl_inst = tuple([int(v) for v in list(wkl_inst)])
            if wkl_inst == shape_tuple:
                return i
        assert False, "{} not found".format(shape_tuple)


def replace_shape_vars(wkl_func_args, shape_vars, new_shape_vars):
    replaced_dyn_args = \
            _ffi_api.ReplaceShapeVars(wkl_func_args, shape_vars, new_shape_vars)
    return tuple(replaced_dyn_args)

def instantiate_dyn_args(wkl_func_args, shape_vars, wkl_inst):
    instantiated_dyn_args = \
            _ffi_api.InstantiateDynArgs(wkl_func_args, shape_vars, wkl_inst)
    return tuple([i.value for i in instantiated_dyn_args])

def serialize_state(state):
    return _ffi_api.SerializeState(state)

def deserialize_state(json_str):
    return _ffi_api.DeserializeState(json_str)
