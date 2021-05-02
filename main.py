"""
Keybert
 * https://github.com/MaartenGr/KeyBERT

Wikimedia search API
 * https://www.mediawiki.org/wiki/API:Search
Looks like max query string length is 300 characters
 * https://phabricator.wikimedia.org/T107947
 * https://en.wikipedia.org/wiki/Help:Searching

TO DO
 * find bug that causes empty keyword list for text 4 with topn=14


"""
import itertools
import logging
import requests
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
):

    logger.info("text: {}".format(text))

    # get keywords from text
    #=====================================================
    keywords = get_keywords(
        model,
        text,
        top_n=keywords_topn,
        ngram_range=keywords_ngram_range,
        diversity=keywords_diversity,
    )

    logger.info("keywords: {}".format(keywords))

    # get wikipedia pages from keywords
    #=====================================================
    keywords_or = " OR ".join([el[0] for el in keywords])
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": keywords_or,
        "srprop": "size|wordcount|timestamp|snippet|sectiontitle|sectionsnippet",
    }
    response = requests.get(url=url, params=params)
    response_json = response.json()
    pages = response_json['query']['search']

    # get wikidata item ids from page ids
    #=====================================================
    page_ids = [el['pageid'] for el in pages]
    page_ids_pipe = "|".join([str(el) for el in page_ids])
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageprops",
        "ppprop": "wikibase_item",
        "redirects": "1",
        "pageids": page_ids_pipe,
    }
    response = requests.get(url=url, params=params)
    response_json = response.json()
    page_to_item = response_json['query']['pages']

    # add wikidata item ids to pages
    #=====================================================
    page_id_to_page = {
        str(page['pageid']): page for page in pages
    }
    for pageid, p2i in page_to_item.items():
        page_id_to_page[pageid]['itemid'] = p2i['pageprops']['wikibase_item']

    # get wikidata items from item ids
    #=====================================================
    item_ids = [el['itemid'] for el in pages]
    item_ids_pipe = "|".join(item_ids)
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "languages": "en",
        "ids": item_ids_pipe,
    }
    response = requests.get(url=url, params=params)
    response_json = response.json()
    items = response_json['entities']

    # add wikidata item ids to pages
    #=====================================================
    item_id_to_page = {
        page['itemid']: page for page in pages
    }
    for itemid, item in items.items():
        item_id_to_page[itemid]['entity'] = item


    return {
        'text': text,
        'keywords': keywords,
        'pages': pages,
        'page_id_to_page': page_id_to_page,
        'page_to_item': page_to_item,
    }

if __name__ == '__main__':

    logging.basicConfig(
        format='[%(funcName)s() ] %(message)s',
        level=logging.INFO)
    sbert_model_name = 'distilbert-base-nli-mean-tokens'
    sentence_transformer_model = SentenceTransformer(
        sbert_model_name, device="cpu")
    keybert_model = KeyBERT(model=sentence_transformer_model)
    wgo = wikimedia_go(keybert_model, text1)
