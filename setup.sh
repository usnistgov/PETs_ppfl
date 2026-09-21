# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software
#
# This file was edited with the assistance of Claude Code (Anthropic, model
# Claude Opus 5). The assistant updated the installed Python version from 3.10
# to 3.12 as part of the Python 3.12 migration, in accordance with the author's
# instructions. All content has been reviewed and verified by the authors.

apt update

add-apt-repository ppa:deadsnakes/ppa

apt update

apt install python3.12

echo | python3.12 --version
