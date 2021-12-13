#!/bin/bash
. .package-version

upload_package()
{
  rm -r -f dist
  rm -r -f build
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