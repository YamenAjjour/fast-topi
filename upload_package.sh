upload_package()
{
  rm -r -f dist
  rm -r -f build
  python3 setup_classifier.py bdist_wheel
  python3 -m twine upload dist/* --
}

increment_version()
{
  VERSION=$((VERSION+1))
  rm -r -f .model-version
  echo "VERSION=${VERSION}" >> .package-version
}