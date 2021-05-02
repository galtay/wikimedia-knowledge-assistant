"""
Wikimedia Knowledge Assistant

Extract keywords from input text and gather relevant wikimedia resources.


Keyword Extraction
==================

Keyword extraction currently done with Keybert
 * https://github.com/MaartenGr/KeyBERT


Wikimedia API resources
=======================

Wikimedia search API
 * https://www.mediawiki.org/wiki/API:Search

Looks like max query string length is 300 characters
 * https://phabricator.wikimedia.org/T107947
 * https://en.wikipedia.org/wiki/Help:Searching

Wikipedia action API action=query
 * https://www.mediawiki.org/wiki/API:Query
 * https://www.mediawiki.org/wiki/API:Page_info_in_search_results
 * https://www.mediawiki.org/wiki/Extension:TextExtracts#Cav


TO DO
=====
* bug in `extract_keywords` causes empty keyword list for text 4 with topn=14
* remove while loop once above is fixed 

"""
import logging
from time import time
from typing import Iterable, List, Tuple

import requests
import keybert


logger = logging.getLogger(__name__)


# add "srsearch": "text you want to search" key value pair 
WP_STATIC_SEARCH_PARAMS = {
    "action": "query",
    "format": "json",
    "list": "search",
    "formatversion": 2,
}

# add "pageids": "pipe seperated string of page ids" key value pair
WP_STATIC_PAGE_INFO_PARAMS = {
    "action": "query",
    "format": "json",
    "formatversion": 2,
    "redirect": 1,
    "prop": "pageprops|pageterms|pageimages|extracts",
    "exintro": 1,
    "explaintext": 1,  # turn this off to get html, but its missing stuff too
}

# add "ids": "pipe seperated string of item ids" key value pair
WD_STATIC_ITEM_PARAMS = {
    "action": "wbgetentities",
    "format": "json",
    "languages": "en",
}


def get_keybert_keywords(
    model: keybert.KeyBERT, 
    text: str, 
    top_n: int=10, 
    ngram_range: Tuple[int, int]=(1, 2),
    diversity: float=0.6
) -> List[Tuple[str, float]]:

    logger.info("extracting keybert keywords from text: % s", text)

    # TODO: remove while loop once empty list bug is figured out
    try_top_n = top_n
    keywords = []
    while len(keywords) == 0:
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words='english',
            use_mmr=True,
            diversity=diversity,
            top_n=try_top_n,
        )
        try_top_n -= 1
    return keywords


def get_google_query_from_keywords(keywords: Iterable[str]):
    keywords_or = " OR ".join(keywords)
    google_query = "{} {}".format("site:en.wikipedia.org", keywords_or)
    return google_query


def wikimedia_go(
        keywords: Iterable[str],
        wp_base_url: str="https://en.wikipedia.org/w/api.php",
        wd_base_url: str="https://www.wikidata.org/w/api.php",
):

    t_start = time()
    logger.info("keywords: % s", keywords)

    # send keywords to wikipedia search api
    # =====================================================
    keywords_or = " OR ".join(keywords)
    wp_search_params = {"srsearch": keywords_or, **WP_STATIC_SEARCH_PARAMS}
    logger.info("querying wikipedia search api using keywords")
    t0 = time()
    wp_search_response = requests.get(url=wp_base_url, params=wp_search_params)
    logger.info("wikipedia search took %.4f seconds", time()-t0)
    wp_search_response_json = wp_search_response.json()
    # TODO: handle continuation results
    assert('batchcomplete' in wp_search_response_json)
    wp_searches = wp_search_response_json['query']['search']

    # get extra information on pages returned by search
    # =====================================================
    page_ids = [el['pageid'] for el in wp_searches]
    page_ids_pipe = "|".join([str(el) for el in page_ids])
    wp_info_params = {"pageids": page_ids_pipe, **WP_STATIC_PAGE_INFO_PARAMS}
    logger.info("querying wikipedia for page info")
    t0 = time()
    wp_info_response = requests.get(url=wp_base_url, params=wp_info_params)
    logger.info("wikipedia info took %.4f seconds", time()-t0)
    wp_info_response_json = wp_info_response.json()
    # TODO: handle continuation results
    assert('batchcomplete' in wp_info_response_json)
    wp_pages = wp_info_response.json()['query']['pages']

    # add url to page info
    #---------------------
    for page in wp_pages:
        page["url"] = "http://en.wikipedia.org/wiki?curid={}".format(page["pageid"])

    # get wikidata items from item ids
    # =====================================================
    item_ids = [el['pageprops']['wikibase_item'] for el in wp_pages if 'pageprops' in el]
    item_ids_pipe = "|".join(item_ids)
    wd_item_params = {"ids": item_ids_pipe, **WD_STATIC_ITEM_PARAMS}
    logger.info("querying wikidata for item info")
    t0 = time()
    wd_response = requests.get(url=wd_base_url, params=wd_item_params)
    logger.info("wikidata took %.4f seconds", time()-t0)
    wd_response_json = wd_response.json()
    wd_entities = wd_response_json['entities']

    # add url to item info
    #---------------------
    for qid, entity in wd_entities.items():
        entity["url"] = "https://www.wikidata.org/wiki/{}".format(entity['id'])


    # make pageid maps and group final output
    # preserve the order in search (sorted by relevance)
    # =====================================================
    pageid_to_item_id = {
        el['pageid']: el['pageprops']['wikibase_item'] 
        for el in wp_pages if 'pageprops' in el}
    pageid_to_wp_pages = {el['pageid']: el for el in wp_pages}

    output = []
    for search in wp_searches:
        pageid = search['pageid']
        entity = wd_entities[pageid_to_item_id[pageid]] if pageid in pageid_to_item_id else None
        element = {
            'wp_search': search,
            'wp_page': pageid_to_wp_pages[pageid],
            'wd_entity': entity,
        }
        output.append(element)

    t_end = time()
    logger.info("total function time %.4f seconds", t_end-t_start)

    return {
        'keywords': keywords,
        'output': output,
    }

