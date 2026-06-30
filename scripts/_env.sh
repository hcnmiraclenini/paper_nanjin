#!/bin/bash
# 项目根目录与 Python 路径（所有 scripts/*.sh 应 source 此文件）
MOE_NANJIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${MOE_NANJIN_ROOT}/src:${MOE_NANJIN_ROOT}/..:${PYTHONPATH}"
cd "${MOE_NANJIN_ROOT}"
