import utils
import string

class Sentence:
    def __init__(self, condition_name='', regions=None, tokens=[], model=None):
        self.regions = [Region(**r) for r in regions]
        self.content = ' '.join(r.content for r in self.regions)
        self.tokens = tokens
        self.region2tokens = self.tokenize_regions(model=model)
        for i, r in enumerate(self.regions):
            r.tokens = self.region2tokens[r.region_number]
            self.regions[i] = r

    def tokenize_regions(self, model=None, eos_tokens=['<eos>', '</S>']):
        """
        Converts self.tokens (list of tokens) to dictionary of 
        <region_number, token list> pairs.
        """
        # initialize current region, content, and counter
        r_idx = 0
        r = self.regions[r_idx]
        content = r.content

        # initialize region-to-token dict
        region_tokens = {r.region_number : [] for r in self.regions}

        # iterate over all tokens in sentence
        for token in self.tokens:

            # remove casing and punctuation for ordered-neurons only
            if model == 'ordered-neurons':
                content = content.lower()
                content = content.translate(
                    str.maketrans('', '', string.punctuation)
                )

            # handle end-of-sentence tokens separately
            if token in eos_tokens:
                region_tokens[r.region_number].append(token)

            # for non-eos tokens
            else:
                # remove leading spaces of current content
                content = content.lstrip()

                # NOTE: quick, untested, dirty hack for roBERTa tokenization
                # deal with token boundaries and decomposition, e.g.
                # This token will decompose. --> ĠThis Ġtoken Ġwill Ġdecom pose .
                if model == 'roberta':
                    if token[0] == 'Ġ':
                        # remove token boundary
                        token = token[1:]

                # if exact match with beginning of content
                if token == content[:len(token)]:
                    region_tokens[r.region_number].append(token)
                    # remove token from content
                    content = content[len(token):]

                    # if end of content (removing spaces), and before last region
                    if content.strip() == '' and r_idx < len(self.regions) - 1:
                        r_idx += 1
                        r = self.regions[r_idx]
                        content = r.content

                # otherwise, move to next region
                else:
                    r_idx += 1
                    r = self.regions[r_idx]
                    content = r.content

        return region_tokens

class Region:
    def __init__(self, region_number=None, content=''):
        utils.save_args(vars())
        self.token_surprisals = []

    def __repr__(self):
        s = 'Region(\n\t{}\n)'.format(str(vars(self)))
        return s

    def agg_surprisal(self, metric):
        self.surprisal = utils.METRICS[metric](self.token_surprisals)
        return self.surprisal
