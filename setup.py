#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = "Mu Yang <http://muyang.pro>"
__copyright__ = "2020 CKIP Lab"
__license__ = "GPL-3.0"

from setuptools import setup, find_namespace_packages
import ckip_transformers as about

################################################################################


def main():

    with open("README.rst", encoding="utf-8") as fin:
        readme = fin.read()

    setup(
        name="ckip-transformers",
        version=about.__version__,
        author=about.__author_name__,
        author_email=about.__author_email__,
        description=about.__description__,
        long_description=readme,
        long_description_content_type="text/x-rst",
        url=about.__url__,
        download_url=about.__download_url__,
        platforms=["linux_x86_64"],
        license=about.__license__,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3 :: Only",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: POSIX :: Linux",
            "Natural Language :: Chinese (Traditional)",
        ],
        python_requires=">=3.6",
        packages=find_namespace_packages(
            include=[
                "ckip_transformers",
                "ckip_transformers.*",
            ]
        ),
        install_requires=[
            "torch>=1.5.0",
            "tqdm>=4.27",
            "transformers>=3.5.0",
        ],
        data_files=[],
    )


################################################################################

if __name__ == "__main__":
    main()
