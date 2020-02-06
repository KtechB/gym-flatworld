from setuptools import setup

setup(name="gym_flatworld",
      version="0.0.1",
      description="simple 2d continuous environment for OpenAI gym",
      authre="KtechB",

      install_requires=["gym", "numpy"],
      test_suite="tests"
      )