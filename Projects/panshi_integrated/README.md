# Sub Processor Command

## atomic_labeling
### atomic label process:
-   input: [./data/] ego.csv, obj.csv, ego_config.json
-   output: ./results/ tracks_result.csv, tracks_meta_result.csv, recording_result.csv
```bash
python -m scripts.atomic_label [ego_path | data/ego.csv] [obj_path | data/obj.csv] [ego_config_path | data/ego_config.json]
```

### atomic label interactive presentation:
-   input: ./results/tracks_result.csv
```bash
streamlit run scripts/atomic_label_plot.py
```

## line_processor
- input: ./data/line.csv
- output: ./results/ line_processed.csv, extracted_line.png
```base
python -m scripts.line_process
```

## TODO: tracks_to_virtual_lane
## TODO: trajectory_extractor
