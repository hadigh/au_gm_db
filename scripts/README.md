Here’s a cleaned-up and GitHub-flavored Markdown version of your notes. I’ve organized it into sections, added proper headings, bullet points, and code formatting where appropriate to improve readability and structure:

---

# 📜 Script Execution Workflow

> **Note:** Check **my revisions 1, 2, and 3** first!

### ✅ Step-by-step Instructions

1. **Run** `WaveformFileVerifier.py`  
   → Verifies if all waveform files exist.

2. **[Optional]** Run `PazFileVerifier.py`  
   → Checks which `pazfile` listed in `stationlist.dat` is missing from `../inputs/paz`.

3. **Run** `WaveformListAugmenter.py`  
   → ⚠️ Modify lines **27** and **32** if needed.

4. **Run** `WaveformListStationListStationVsMerger.py`  
   → ⚠️ Modify lines **55–63**.

5. **Run** `GMInputGenerator.py`  
   → ⚠️ Modify line **38**.  
   → Had to run in **two segments**:
   - Until end of 2022
   - From start of 2023 to end (see line 181)

6. **Run** `PortalStationxmlGenerator.py`  
   → ⚠️ Modify lines **271** and **275**.

7. After running `gmrecords assemble`, run **Assembly Validator**.

---

# 🔁 My Revisions

### 1. `fix_channel_codes.py`  
- Manually assigned correct channel codes to problematic streams.  
- Saved corrected MiniSEED files in `wf_data`.  
- ⚠️ If replaced from NAS, this must be redone.

### 2. Manual waveform downloads (after correspondence with Robert Pickle, ANU):  
- Fixed incorrect sampling rate (250 Hz on vertical components) for:
  - `2023-02-05T00.35.WG.WBP10.mseed`
  - `2022-12-11T14.30.WG.WBP10.mseed`
  - `2023-01-05T05.08.WG.WBP10.mseed`

### 3. Manual edits to `wf_lst.csv`:
- **BRAT** was wrongly assigned to **AUMAG**  
  → Correct: `2023-06-29T15.28.S1.AUMAG.mseed`
- **MBWA** was wrongly assigned to **NWAO**  
  → Correct: `2002-06-23T11.19.IU.NWAO.mseed`

#### MUN station fixes:
- Added to `new_hsd` and referenced in `20250130_updated_au_wf_lst4ghd_mun_fix.csv`:
  - `2022-01-22T07.55.AU.MUN.mseed`
  - `2022-01-24T21.22.AU.MUN.mseed`
  - `2022-01-24T21.48.AU.MUN.mseed`
  - `2022-02-01T10.39.AU.MUN.mseed`

---

## 🛠️ Fixes by Eric @ USGS

### 4. `waveform_metric_calculator.py`  
Path:  
`/home/hadi/miniconda3/envs/gmprocess/lib/python3.9/site-packages/gmprocess/metrics/waveform_metric_calculator.py`

```python
#### HG
ss = self.stream
if ss.num_horizontal < 2:
    filtered_dict = {
        key: value for key, value in self.steps.items() if "rotd" not in key
    }

for metric, metric_steps in filtered_dict.items():
```

---

### 5. `flatfile.py`  
Path:  
`/home/hadi/miniconda3/envs/gmprocess/lib/python3.9/site-packages/gmprocess/io/asdf/flatfile.py`

```python
## HG
# with open(default_config_file, "r", encoding="utf-8") as f:
#     yaml = YAML()
#     yaml.preserve_quotes = True
#     default_config = yaml.load(f)
# update_dict(self.workspace.config, default_config)
```

---

### 6. `HumanReviewGUI.py`  
→ Add the following at **line 350**:

```python
if tr.passed:
```

---

# ⚠️ Notes & Issues

- **YAML spacing:** Use **spaces**, not **tabs**, in `config.yml` to avoid strange errors.
- **Unknown issue example:**

```python
f = '../gmprocess_projects/data/20200415071104/raw/AU.CNB..BHE__2020-04-15T07:07:00.019538Z__2020-04-15T07:36:59.994538Z.mseed'
```
- **Ignored noise files:**
- `1994-08-06T11.05.MEL.PIN.mseed`


- **Config flags:**

```yaml
any_trace_failures: False

check_instrument:
  n_max: 3
  n_min: 1
  require_two_horiz: False

check_clipping:
  threshold: 1.0
```

---

# 📦 Network & Channel Fixes

### 5. Revised "1P" network dataless seed file using:
- `anu_response_reviser.py`
- `dev_inv_anu_tmp.py`
- `merge_ANU_II_IU_S1.py`

### 6. Updated channel codes for problematic stations:
- Identified via `WaveformListAugmenter.py`
- Fixed using `fix_channel_codes.py`


---

# 🧠 Hints & Misc

- Manually revised `strec` config to point to the correct directory.
- Found duplicate lines in `stationlist.dat` for **UM** network (e.g., WV).

---
