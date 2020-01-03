import inquirer
import numpy as np
import requests
from pandas import read_html

from src.helper import log


def _find_tables(url):
    """
    Parse all <table> tags found on the page located at ``url``
    :param url: Location of the webpage to be parsed
    :return: List of dicts, each dict containing the heads and rows of the table
    """
    return [
        {
            'head': table.columns.to_numpy(),
            'rows': table.to_numpy()
        }
        for table in read_html(str(requests.get(url).text))
    ]


def _select_table(tables):
    """
    Ask the user to select one of the tables
    :param tables: The tables to select from
    :return: The selected table
    """
    questions = [
        inquirer.List('table',
                      message='Select table to crawl',
                      choices=[f'[{i + 1}] {" | ".join(table["head"])}' for i, table in enumerate(tables)],
                      ),
    ]
    result = inquirer.prompt(questions)
    index = int(result['table'].split(']')[0].strip('[')) - 1
    return tables[index]


def _select_column(table):
    """
    Ask the user to select a column of the selected table
    :param table: The selected table
    :return: The index of the selected column
    """
    questions = [
        inquirer.List('column',
                      message='Select column to crawl',
                      choices=[head for head in list(table['head'])],
                      ),
    ]
    result = inquirer.prompt(questions)
    index = np.where(table['head'] == result['column'])[0][0]
    return index


def _crawl(url, outfile):
    """
    Main function of the file. Crawl terms webpage located at ``url``, and save them to ``outfile``
    :param url: The url to crawl from
    :param outfile: The file to save the crawled term to
    """
    tables = _find_tables(url)
    table = _select_table(tables)
    column = _select_column(table)
    result = table['rows'][:, column]
    with open(outfile, 'w') as file:
        file.write('\n'.join([str(r) for r in result]))
    log(f'Saved {len(result)} terms to {outfile}')


def parse_input(args):
    """
    Parse commandline arguments and start crawl
    :param args: Commandline arguments
    """
    url = args.url
    outfile = args.output
    _crawl(url, outfile)
