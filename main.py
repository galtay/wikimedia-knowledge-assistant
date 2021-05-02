"""
Keybert
 * https://github.com/MaartenGr/KeyBERT

Wikimedia search API
 * https://www.mediawiki.org/wiki/API:Search

Looks like max query string length is 300 characters
 * https://phabricator.wikimedia.org/T107947
 * https://en.wikipedia.org/wiki/Help:Searching

Wikipedia action API action=query
* https://www.mediawiki.org/wiki/API:Query

https://www.mediawiki.org/wiki/API:Page_info_in_search_results
https://www.mediawiki.org/wiki/Extension:TextExtracts#Cav

TO DO
 * find bug that causes empty keyword list for text 4 with topn=14


"""
import logging
import requests
from time import time
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


text1 = """
Researchers are trying to determine whether an 18,000-year-old puppy found in
Siberia is a dog or a wolf. The canine - which was two months old when it died
 - has been remarkably preserved in the permafrost of the Russian region, with
its fur, nose and teeth all intact. DNA sequencing has been unable to determine
the species. Scientists say that could mean the specimen represents an
evolutionary link between wolves and modern dogs.
"""

text2 = """
U.S. intelligence cannot say conclusively that Saddam Hussein
has weapons of mass destruction, an information gap that is complicating
White House efforts to build support for an attack on Saddam's Iraqi regime.
The CIA has advised top administration officials to assume that Iraq has
some weapons of mass destruction.  But the agency has not given President
Bush a "smoking gun," according to U.S. intelligence and administration
officials.
"""

text3 = """
The development of T-cell leukaemia following the otherwise
successful treatment of three patients with X-linked severe combined
immune deficiency (X-SCID) in gene-therapy trials using haematopoietic
stem cells has led to a re-evaluation of this approach.  Using a mouse
model for gene therapy of X-SCID, we find that the corrective therapeutic
gene IL2RG itself can act as a contributor to the genesis of T-cell
lymphomas, with one-third of animals being affected.  Gene-therapy trials
for X-SCID, which have been based on the assumption that IL2RG is minimally
oncogenic, may therefore pose some risk to patients.
"""

text4 = """
Share markets in the US plummeted on Wednesday, with losses accelerating
after the World Health Organization declared the coronavirus outbreak a pandemic.
"""


WP_SEARCH_PARAMS = {
    "action": "query",
    "format": "json",
    "list": "search",
    "formatversion": 2,
}

WP_PAGE_INFO_PARAMS = {
    "action": "query",
    "format": "json",
    "formatversion": 2,
    "redirect": 1,
    "prop": "pageprops|pageterms|pageimages|extracts",
    "exintro": 1,
    "explaintext": 1,  # turn this off to get html, but its missing stuff too
}

WD_ITEM_PARAMS = {
    "action": "wbgetentities",
    "format": "json",
    "languages": "en",
}


def get_keywords(model, text, top_n=10, ngram_range=(1, 2), diversity=0.6):

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


def get_google_query_from_keywords(keywords):
    keywords_or = " OR ".join([el[0] for el in keywords])
    google_query = "{} {}".format("site:en.wikipedia.org", keywords_or)
    return google_query


def wikimedia_go(
        model,
        text,
        keywords_topn=10,
        keywords_ngram_range=(1, 2),
        keywords_diversity=0.6,
        wp_base_url="https://en.wikipedia.org/w/api.php",
        wd_base_url="https://www.wikidata.org/w/api.php",
):

    t_start = time()
    logger.info("model: % s", model)
    logger.info("text: % s", ' '.join(text.split('\n')))

    # get keywords from text
    # =====================================================
    t0 = time()
    keywords = get_keywords(
        model,
        text,
        top_n=keywords_topn,
        ngram_range=keywords_ngram_range,
        diversity=keywords_diversity,
    )
    logger.info("extracting keywords took %.4f seconds", time()-t0)
    logger.info("keywords: % s", keywords)

    # send keywords to wikipedia search api
    # =====================================================
    keywords_or = " OR ".join([el[0] for el in keywords])
    wp_search_params = {"srsearch": keywords_or, **WP_SEARCH_PARAMS}
    logger.info("querying wikipedia search api using keywords")
    t0 = time()
    wp_search_response = requests.get(url=wp_base_url, params=wp_search_params)
    logger.info("wikipedia search took %.4f seconds", time()-t0)
    wp_search_response_json = wp_search_response.json()
    assert('batchcomplete' in wp_search_response_json)
    wp_search = wp_search_response_json['query']['search']

    # get extra information on pages returned by search
    # =====================================================
    page_ids = [el['pageid'] for el in wp_search]
    page_ids_pipe = "|".join([str(el) for el in page_ids])
    wp_info_params = {"pageids": page_ids_pipe, **WP_PAGE_INFO_PARAMS}
    logger.info("querying wikipedia for page info")
    t0 = time()
    wp_info_response = requests.get(url=wp_base_url, params=wp_info_params)
    logger.info("wikipedia info took %.4f seconds", time()-t0)
    wp_info_response_json = wp_info_response.json()
    assert('batchcomplete' in wp_info_response_json)
    wp_pages = wp_info_response.json()['query']['pages']

    # get wikidata items from item ids
    # =====================================================
    item_ids = [el['pageprops']['wikibase_item'] for el in wp_pages]
    item_ids_pipe = "|".join(item_ids)
    wd_item_params = {"ids": item_ids_pipe, **WD_ITEM_PARAMS}
    logger.info("querying wikidata for item info")
    t0 = time()
    wd_response = requests.get(url=wd_base_url, params=wd_item_params)
    logger.info("wikidata took %.4f seconds", time()-t0)
    wd_response_json = wd_response.json()
    entities = wd_response_json['entities']

    # make pageid maps and group final output
    # preserve the order in search (sorted by relevance)
    # =====================================================
    pageid_to_item_id = {el['pageid']: el['pageprops']['wikibase_item'] for el in wp_pages}
    pageid_to_wp_pages = {el['pageid']: el for el in wp_pages}

    output = []
    for search in wp_search:
        element = {
            'search': search,
            'page': pageid_to_wp_pages[search['pageid']],
            'entity': entities[pageid_to_item_id[search['pageid']]],
        }
        output.append(element)

    logger.info("finished")

    return {
        'text': text,
        'keywords': keywords,
        'output': output,
        'wp_search': wp_search,
        'wp_pages': wp_pages,
        'entities': entities,
    }

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    sbert_model_name = 'distilbert-base-nli-mean-tokens'
    sentence_transformer_model = SentenceTransformer(
        sbert_model_name, device="cpu")
    keybert_model = KeyBERT(model=sentence_transformer_model)
    wgo = wikimedia_go(keybert_model, text1)
