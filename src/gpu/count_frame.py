#!/usr/bin/env python3
"""Report the frame's recorded command count for each fusion toggle.

A diagnostic for the OpenCL port: `test_joint_qkv.frames` asserts the epilogue saves
`210 + folded` commands, and the count depends on which composed ops the backend
records as one command. This prints the totals so the two backends can be compared
(`NR_GPU_BACKEND=opencl` vs the Vulkan default).
"""
import pathlib
import sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ref'))


def main():
    import nr_frame
    backend = nr_frame.ResidentBackend()
    try:
        rt = backend.runtime
        for height, width in ((320, 320),):
            color = np.random.default_rng(70).random((height, width, 3), dtype=np.float32)
            features = nr_frame.make_features(color, **nr_frame.PROFILES['standard'])
            frame = backend.frame(height, width)
            rt.specialize(7)
            rt.fuse_qk = True
            rt.joint_qkv = False
            rt.qkv_epilogue = False
            counts = {}
            for name, joint, epi in (('baseline', False, False), ('joint', True, False),
                                     ('epilogue', True, True)):
                rt.joint_qkv = joint
                rt.qkv_epilogue = epi
                passes = []
                frame.run(features, execution='replay', submits=passes)
                counts[name] = passes[0]
            print(f"counts={counts} joint_saves={counts['baseline'] - counts['joint']} "
                  f"epi_saves={counts['baseline'] - counts['epilogue']}")
            print(f'window_gather={rt.lib.xmx_window_gather()} '
                  f'fuse_partition={rt.fuse_partition} fuse_window_block={rt.fuse_window_block} '
                  f'fuse_pool={rt.fuse_pool} fuse_head={rt.fuse_head}')
    finally:
        backend.close()


if __name__ == '__main__':
    main()