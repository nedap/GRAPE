### PartNet Dataset: 
The Partnet Dataset is structured in the following manner:


```
PartNet/
├── data_v0/
│   ├── 1/
│   │   ├── obj/
│   │   │   ├── new-0.obj
│   │   │   └── original-1.obj
│   │   ├── parts_render/
│   │   │   ├── 0.png
│   │   │   └── 0.txt
│   │   ├── parts_render_after_merging/
│   │   │   ├── 0.png
│   │   │   └── 0.txt
│   │   ├── point_sample/
│   │   │   ├── label-10000.txt
│   │   │   ├── pts-10000.txt
│   │   │   ├── pts-10000.ply
│   │   │   ├── pts-10000.pts
│   │   │   ├── sample-points-all-label-10000.txt
│   │   │   ├── sample-points-all-pts-label-10000.ply
│   │   │   ├── sample-points-all-pts-nor-rgba-10000.ply
│   │   │   └── sample-points-all-pts-nor-rgba-10000.txt
│   │   ├── meta.json
│   │   ├── result_after_merging.json
│   │   ├── result.json
│   │   ├── tree_hier_after_merging.html
│   │   └── tree_hier.html
│   ...
└── 26671/
    ...

```
Explanation of each file (from [PartNet Repo](https://github.com/daerduoCarey/partnet_dataset/tree/master))
```
data/                                       # Download PartNet data from Google Drive and unzip them here
    42/
        result.json                     # A JSON file storing the part hierarchical trees from raw user annotation
        result_after_merging.json       # A JSON file storing the part hierarchical trees after semantics merging (the final data)
        meta.json                       # A JSON file storing all the related meta-information
        objs/                           # A folder containing several part obj files indexed by `result.json`
                                        # Note that the parts here are not the final parts. Each individual obj may not make sense.
                                        # Please refer to `result.json` and read each part's obj files. Maybe many obj files make up one part.
            original-*.obj              # Indicate this is an exact part mesh from the original ShapeNet model
            new-*.obj                   # Indicate this is a smoothed and cut-out part mesh in PartNet annotation cutting procedure
        tree_hier.html                  # A simple HTML visualzation for the hierarchical annotation (before merging)
        part_renders/                   # A folder with rendered images supporting `tree_hier.html` visualization
        tree_hier_after_merging.html    # A simple HTML visualzation for the hierarchical annotation (after merging)
        part_renders_after_merging/     # A folder with rendered images supporting `tree_hier_after_merging.html` visualization
        point_sample/                   # We sample 10,000 points for point cloud learning
            pts-10000.txt                               # point cloud directly sampled from the combination of part meshes under `objs/`
            label-10000.txt                             # the labels are the id in `result.json`
            sample-points-all-pts-nor-rgba-10000.txt    # point cloud directly sampled from the whole ShapeNet model with labels transferred from `label-10000.txt`
            sample-points-all-label-10000.txt           # labels propagated to `sample-points-all-pts-nor-rgba-10000.txt`
```
Download: You can download the processed data from [PartNet Dataset](https://github.com/daerduoCarey/partnet_dataset) repository, or download from the [official website](https://shapenet.org/download/parts) and process it by yourself.
