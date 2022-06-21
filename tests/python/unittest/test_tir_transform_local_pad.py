# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm.testing


def sample_dense_sched(sch):
    """
    
    """
    b0 = sch.get_block(name="update", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l2, l3, l4 = sch.get_loops(block=b0)
    l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[None, 1, 8, 8, 2])
    l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[None, 2, 32, 1, 2])
    l28, l29, l30 = sch.split(loop=l4, factors=[None, 1, 4])
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 = sch.fuse(l10, l20)
    sch.bind(loop=l31, thread_axis="blockIdx.x")
    l32 = sch.fuse(l11, l21)
    sch.bind(loop=l32, thread_axis="vthread.x")
    l33 = sch.fuse(l12, l22)
    sch.bind(loop=l33, thread_axis="threadIdx.x")
    b34 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b34, loop=l33, preserve_unit_loops=True)
    b35 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b35, loop=l28, preserve_unit_loops=True)
    l36, l37, l38, l39, l40, l41 = sch.get_loops(block=b35)
    l42 = sch.fuse(l40, l41)
    v43 = 3
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b44, loop=l28, preserve_unit_loops=True)
    l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b44)
    l51 = sch.fuse(l49, l50)
    v52 = 4
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v52)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=512)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch")
    l54, l55, l56, l57, l58 = sch.get_loops(block=b35)
    l59, l60, l61 = sch.split(loop=l58, factors=[None, 256, v43])
    sch.vectorize(loop=l61)
    sch.bind(loop=l60, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch")
    l61, l62, l63, l64, l65 = sch.get_loops(block=b44)
    l66, l67, l68 = sch.split(loop=l65, factors=[None, 256, v52])
    sch.vectorize(loop=l68)
    sch.bind(loop=l67, thread_axis="threadIdx.x")
    b69 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b69, ann_key="meta_schedule.unroll_explicit")
    b70, b71, b72, b73 = sch.get_child_blocks(b69)
    l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b70)
    sch.annotate(block_or_loop=l74, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l74, ann_key="pragma_unroll_explicit", ann_val=1)
    l80, l81, l82, l83, l84, l85, l86 = sch.get_loops(block=b71)
    sch.annotate(block_or_loop=l80, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l80, ann_key="pragma_unroll_explicit", ann_val=1)
    l87, l88, l89, l90, l91, l92, l93, l94, l95, l96 = sch.get_loops(block=b72)
    sch.annotate(block_or_loop=l87, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l87, ann_key="pragma_unroll_explicit", ann_val=1)
    l97, l98, l99, l100, l101 = sch.get_loops(block=b73)
    sch.annotate(block_or_loop=l97, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l97, ann_key="pragma_unroll_explicit", ann_val=1)
    b102 = sch.get_block(name="update", func_name="main")
    l103, l104, l105, l106, l107, l108, l109, l110, l111, l112 = sch.get_loops(block=b102)
    b113 = sch.decompose_reduction(block=b102, loop=l106)


if __name__ == "__main__":
    test_dense()
