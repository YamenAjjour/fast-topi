#!/bin/bash
. .package-version

upload_package()
{
  python3 setup.py bdist_wheel
  python3 -m twine upload dist/* --
}

increment_version()
{
  echo $VERSION
  VERSION=$((VERSION+1))

  rm -r -f .package-version
  echo "VERSION=${VERSION}" >> .package-version
}
upload_package
increment_version