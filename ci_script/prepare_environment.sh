#!/usr/bin/env bash

: "${DTK_PKG:?DTK_PKG must point to the DTK archive mounted on the Runner}"
: "${TORCH_VERSION:?TORCH_VERSION must be set}"

cp -f "${DTK_PKG}" /opt/
cd /opt
tar -xzf "$(basename "${DTK_PKG}")"
rm -rf dtk
mv dtk-* dtk
source /opt/dtk/env.sh

cd "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set by GitLab}"

# Install the newest AICC package available on the Runner's shared storage.
# Its absence is non-fatal because some Runners include AICC already.
latest_aicc_run="$(ls /ArchivedFile/ai_cc/nightly/dtk_llvm_*.run 2>/dev/null | sort -t '_' -k4 -nr | head -n1)"
if [[ -n "${latest_aicc_run}" ]]; then
  aicc_installer="$(basename "${latest_aicc_run}")"
  cp "${latest_aicc_run}" "/opt/${aicc_installer}"
  cd /opt
  chmod +x "${aicc_installer}"

  # GitLab Runner enables pipefail.  When the installer exits, `yes` may get
  # SIGPIPE and make the whole pipeline look failed even though installation
  # succeeded.  Temporarily disable pipefail and keep the installer's status.
  set +o pipefail
  yes | bash "${aicc_installer}"
  aicc_status="${PIPESTATUS[1]}"
  set -o pipefail
  if [[ "${aicc_status}" -ne 0 ]]; then
    echo "AICC installation failed: ${aicc_installer}" >&2
    exit "${aicc_status}"
  fi

  echo "AICC installation completed: ${aicc_installer}"
else
  echo "No AICC installer found; continuing with the existing environment."
fi

cd "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set by GitLab}"

pip3 install pyyaml -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
pip3 install hypothesis -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
pip3 install Ninja -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
pip3 install MonkeyType -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple                        
pip3 install wheel -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
pip3 install packaging -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
pip3 install einops -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
pip3 install pytest -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
pip3 install setuptools  --force-reinstall --no-deps -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
python -m pip install torch=="${TORCH_VERSION}" triton ciupload auditwheel patchelf cismi

export PYTHON_VERSION="$(python --version 2>&1 | awk -F '[ .]' '{print $2 "." $3}')"

declare -A NUMPY_VERSIONS=(
  ["3.8"]="1.21.6"
  ["3.9"]="1.22.4"
  ["3.10"]="1.24.3"
  ["3.11"]="1.26.2"
  ["3.12"]="1.26.2"
  ["3.13"]="2.1.2"
  ["3.14"]="2.3.4"
)

NUMPY_VERSION="${NUMPY_VERSIONS[$PYTHON_VERSION]}"
if [[ -z "${NUMPY_VERSION}" ]]; then
  echo "Warning: no numpy version configured for Python ${PYTHON_VERSION}; using numpy==1.24.3"
  NUMPY_VERSION="1.24.3"
fi

echo "Installing numpy==${NUMPY_VERSION} ..."
python -m pip uninstall -y numpy 2>/dev/null || true
python -m pip install numpy=="${NUMPY_VERSION}" \
  --force-reinstall \
  --no-deps \
  --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/

python -c "import numpy; print(f'numpy {numpy.__version__} installed successfully')"

python -m pip install optest -i http://10.16.1.201:9929/nightly/dtk2604/+simple/ --trusted-host 10.16.1.201 --force-reinstall

hipcc --version
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip)"
