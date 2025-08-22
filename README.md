# Papers Scrapper Agent

This agent was presented in the Tribu AI context as a showcase of the agents potential without graph frameworks such as LangGraph.

* The repository behaves as follows:
* Reads a research paper from an input file
* Identifies the main dataset used in the paper
* Searches for it on Kaggle
* Downloads the best matching option
* Explores the dataset (currently limited to CIFAR-10 for MVP purposes)
* Generates code templates and a paper-to-code wiki
* Renders static code fields with hyperparameters extracted

## Installation

In order to run the script, you'll need to install dependencies

`
pip install -r requirements.txt
`

After that, you'll need to install the main package papers2code

`
pip install .
`

## Usage

Update the .env.template to .env, add your OPENAI_API_KEY, or directly export it:

`
export OPENAI_API_KEY=your-api-key
`

Run the main script:

`
python scripts/run_paper_agent.py --paper "files/wide_resnet_paper.pdf" --out "artifacts"
`

Where --paper is the path of your paper and --out is the folder where you'll save your static

## Outputs

The agent generates the following artifacts:

```
├── candidates.json
├── code
│   ├── DATASET_CARD_TEMPLATE.md
│   ├── Makefile
│   ├── README.md
│   ├── config.yaml
│   ├── environment.yml
│   ├── notebooks
│   └── src
├── dataset_card.md
├── dataset_quanbk_cifar10
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   └── test_batch
├── eda
│   ├── class_counts.png
│   └── sample_grid.png
├── images_sample
│   ├── airplane
│   ├── automobile
│   ├── bird
│   ├── cat
│   ├── deer
│   ├── dog
│   ├── frog
│   ├── horse
│   ├── ship
│   └── truck
├── logs
│   ├── mentions_prompt.txt
│   ├── mentions_response.parsed.json
│   ├── mentions_response.raw.json
│   ├── methods_prompt.txt
│   ├── methods_response.parsed.json
│   └── methods_response.raw.json
├── method_spec.json
├── paper_to_code_wiki.md
├── resolver_matches.json
└── selection.json
```

## TL;DR

The current development is on MVP version. Partial results are defined under specific behaviors and data pipelines. MVP was based on the [Wide Resnet Paper](https://arxiv.org/abs/1605.07146)

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0)

```
Copyright (C) 2025 Thefrancho

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```