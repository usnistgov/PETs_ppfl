# For licensing matters, please refer to the licensing statement at:
# https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications#software

apt update

add-apt-repository ppa:deadsnakes/ppa

apt update

apt install python3.10

echo | python3.10 --version
