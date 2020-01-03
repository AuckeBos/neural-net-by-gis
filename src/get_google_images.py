import argparse
import os
from datetime import date

from dateutil.relativedelta import relativedelta
from icrawler.builtin import GoogleImageCrawler

from src.helper import get_progress, log

max_per_iteration = 1000


def _get_date_filter(end_date):
    """
    Create datefilter the formated needed for GoogleImageCrawler, ending at ``end_date``; starting a year earlier
    :param end_date: The end date of the filter
    :return: Datefilter in correct format
    """
    start_date = end_date + relativedelta(years=-1)
    return (start_date.year, start_date.month, start_date.day), (end_date.year, end_date.month, end_date.day)


def _download(query, dir, amount_to_crawl):
    """
    Download ``amount_to_crawl`` images from Google Image Serach for query ``query``, save in directory ``dir``
    :param query: Search query for Google Image Search
    :param dir: Directory to save the results in
    :param amount_to_crawl:  Number of pictures to crawl
    """
    intlen = len(str(amount_to_crawl))
    google_crawler = GoogleImageCrawler(
        feeder_threads=10,
        parser_threads=1,
        log_level=100,
        downloader_threads=10,
        storage={'root_dir': dir})
    end_date = date.today()
    amount_crawled = 0
    while amount_crawled < amount_to_crawl > 0:
        crawling = min(max_per_iteration, amount_to_crawl - amount_crawled)
        date_filter = _get_date_filter(end_date)
        log(f'{get_progress((amount_crawled + crawling) / amount_to_crawl)} '
            f'crawling images '
            f'{amount_crawled:0{intlen}d} - {(amount_crawled + crawling):0{intlen}d} / {amount_to_crawl} '
            f'for \'{query}\''
            f' in daterange {date_filter}', end='\r'
            )
        google_crawler.crawl(keyword=query,
                             filters={'date': date_filter},
                             max_num=crawling,
                             file_idx_offset='auto')
        amount_crawled += crawling
        end_date += relativedelta(years=-1)
    print('')


def parse_input(args: argparse.Namespace):
    """
    Parse commandline arguments and download images
    :param args: Commandline arguments
    """
    input_file = args.input
    output_dir = args.output.rstrip(os.sep)
    amount = args.count
    with open(input_file, 'r') as types:
        items = types.read().splitlines()
    for item in items:
        itemdir = f'{output_dir}{os.sep}{item}'
        if not os.path.exists(itemdir):
            os.makedirs(itemdir)
        _download(item, itemdir, amount)
