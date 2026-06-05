#!/usr/bin/env bash
# Run MegaGDN PTO A5 kernels under CANN Simulator (cannsim record, Ascend950).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A5_DIR="${SCRIPT_DIR}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/bin/setenv.bash"

export MEGAGDN_PTO_ARCH="${MEGAGDN_PTO_ARCH:-a5}"
export GDN_NPU_DEVICE="${GDN_NPU_DEVICE:-npu:0}"

SOC="${CANNSIM_SOC:-Ascend950}"
OUTPUT_DIR="${A5_DIR}/outputs/cannsim"
mkdir -p "${OUTPUT_DIR}"
ulimit -n 65535

cd "${A5_DIR}"

USER_OPTS="${*}"
if [[ -z "${USER_OPTS}" ]]; then
  USER_OPTS="--stage cumsum --mode bench --n-seq 1 --l-seg 128 --H 16 --Hg 16"
fi

echo "==> cannsim record (${SOC})"
echo "    ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "    user options: ${USER_OPTS}"

chmod +x "${A5_DIR}/run_cannsim_entry.sh"
cannsim record \
  -s "${SOC}" \
  -o "${OUTPUT_DIR}" \
  "${A5_DIR}/run_cannsim_entry.sh" \
  -u "${USER_OPTS}"
CANNSIM_RC=$?

OUT_JSON=""
if [[ "${USER_OPTS}" == *"--output-json"* ]]; then
  OUT_JSON="$(python3 - "${USER_OPTS}" <<'PY'
import shlex, sys
args = shlex.split(sys.argv[1])
for i, a in enumerate(args):
    if a == "--output-json" and i + 1 < len(args):
        print(args[i + 1])
        break
PY
)"
fi

if [[ -n "${OUT_JSON}" && -f "${OUT_JSON}" ]]; then
  LATEST_EXPORT="$(ls -td "${OUTPUT_DIR}"/cannsim_* 2>/dev/null | head -1 || true)"
  if [[ -n "${LATEST_EXPORT}" && -d "${LATEST_EXPORT}/log_ca" ]]; then
    python3 - "${OUT_JSON}" "${LATEST_EXPORT}/log_ca" <<'PY'
import json, re, sys
from pathlib import Path

out_json, log_ca = Path(sys.argv[1]), Path(sys.argv[2])
ghz = float(__import__("os").environ.get("CANNSIM_AICORE_GHZ", "1.8"))
pat = re.compile(r"start:\s*(\d+),\s*tick:\s*(\d+)")
spans = []
for dump in sorted(log_ca.glob("*.instr_log.dump")):
    for line in dump.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            spans.append(int(m.group(2)) - int(m.group(1)))
if spans:
    data = json.loads(out_json.read_text())
    row = data["results"][0]
    if not row.get("hw_predicted_ms"):
        row["hw_predicted_ms"] = max(spans) / (ghz * 1_000_000.0)
        row["hw_predicted_source"] = "log_ca_cycles"
        out_json.write_text(json.dumps(data, indent=2) + "\n")
PY
  fi
fi

if [[ "${CANNSIM_RC}" -ne 0 ]] && [[ -n "${OUT_JSON}" && -f "${OUT_JSON}" ]]; then
  if python3 - "${OUT_JSON}" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
rows = data.get("results", [])
sys.exit(0 if rows and rows[0].get("sim_wall_s") is not None else 1)
PY
  then
    echo "==> cannsim exited ${CANNSIM_RC} but ${OUT_JSON} is valid; treating as success"
    exit 0
  fi
fi
exit "${CANNSIM_RC}"
