import logging
from time import time
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from wikimedia_knowledge_assistant import get_keybert_keywords, wikimedia_go


logger = logging.getLogger(__name__)


text1 = (
    "Researchers are trying to determine whether an 18,000-year-old puppy "
    "found in Siberia is a dog or a wolf. The canine - which was two months "
    "old when it died - has been remarkably preserved in the permafrost of "
    "the Russian region, with its fur, nose and teeth all intact. DNA "
    "sequencing has been unable to determine the species. Scientists say that "
    "could mean the specimen represents an evolutionary link between wolves "
    "and modern dogs."
)

text2 = (
    "U.S. intelligence cannot say conclusively that Saddam Hussein "
    "has weapons of mass destruction, an information gap that is "
    "complicating White House efforts to build support for an attack on "
    "Saddam's Iraqi regime. The CIA has advised top administration officials "
    "to assume that Iraq has some weapons of mass destruction.  But the agency "
    "has not given President Bush a 'smoking gun,' according to U.S. "
    "intelligence and administration officials."
)

text3 = (
    "The development of T-cell leukaemia following the otherwise "
    "successful treatment of three patients with X-linked severe combined "
    "immune deficiency (X-SCID) in gene-therapy trials using haematopoietic "
    "stem cells has led to a re-evaluation of this approach.  Using a mouse "
    "model for gene therapy of X-SCID, we find that the corrective "
    "therapeutic gene IL2RG itself can act as a contributor to the genesis of "
    "T-cell lymphomas, with one-third of animals being affected.  Gene-therapy "
    "trials for X-SCID, which have been based on the assumption that IL2RG is "
    "minimally oncogenic, may therefore pose some risk to patients."
)

text4 = (
    "Share markets in the US plummeted on Wednesday, with losses accelerating "
    "after the World Health Organization declared the coronavirus outbreak a "
    "pandemic."
)

texts = [text1, text2, text3, text4]


def make_report(wgo, text):
    print("====== wgo and text report ======")
    print("------ text ------")
    print(text, end='\n\n')
    print("------ keywords ------")
    print(wgo['keywords'], end='\n\n')
    print("------ pages ------")
    for out in wgo['output']:
        print(out['wp_page']['title'])
        print(out['wp_page']['url'])
        if out['wd_entity'] is not None:
            print(out['wd_entity']['url'])
        print()



if __name__ == '__main__':

    #logging.basicConfig(level=logging.INFO)


    # load model
    # =====================================================
    model_name = 'distilbert-base-nli-mean-tokens'
    sentence_transformer_model = SentenceTransformer(model_name, device="cpu")
    keybert_model = KeyBERT(model=sentence_transformer_model)

    # get wikimedia info
    # =====================================================
    keywords_topn = 10
    keywords_ngram_range = (1,2)
    keywords_diversity = 0.6

    wgos_and_text = []
    for text in texts:

        t0 = time()
        keywords_and_scores = get_keybert_keywords(
            keybert_model,
            text,
            top_n=keywords_topn,
            ngram_range=keywords_ngram_range,
            diversity=keywords_diversity,
        )
        logger.info("extracting keywords took %.4f seconds", time()-t0)

        keywords = [el[0] for el in keywords_and_scores]
        wgo = wikimedia_go(keywords)
        wgos_and_text.append((wgo, text))


    # get wikimedia info
    # =====================================================
    for wgo, text in wgos_and_text:
        make_report(wgo, text)
