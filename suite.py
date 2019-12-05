import utils
import re

punct_re = re.compile(r"^[-.?!,\*]+$")

UNCASED_MODELS = ['ordered_neurons', 'ngram']
BPE_MODELS = ['gpt-2', 'roberta']

class Sentence:
    def __init__(self, condition_name='', regions=None, 
                 tokens=[], unks=[], model=None):
        self.regions = [Region(**r) for r in regions]
        self.content = ' '.join(r.content for r in self.regions)
        self.tokens = tokens
        self.unks = unks
        # TODO: require containers to specify casing scheme of model
        self.region2tokens = self.tokenize_regions(
            model=model, uncased=(model in UNCASED_MODELS), bpe=(model in BPE_MODELS)
        )
        for i, r in enumerate(self.regions):
            r.tokens = self.region2tokens[r.region_number]
            self.regions[i] = r

    def tokenize_regions(self, model=None, uncased=False, nopunct=False, 
                         bpe=False, eos_tokens=['<eos>', '</S>', '</s>']):
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
        for t_idx, token in enumerate(self.tokens):
            # handle end-of-sentence tokens separately
            if token in eos_tokens:
                region_tokens[r.region_number].append(token)
                break
            # handle out-of-vocabulary tokens separately
            # elif self.unks[t_idx] == 1:
                # raise RuntimeError(
                #     '"{}" is out-of-vocabulary for {}'.format(token, model)
                # )

            # HACK: remove casing and punctuation
            if uncased:
                content = content.lower()
            if nopunct:
                content = ' '.join([s for s in content.split(' ') if not punct_re.match(s)])

            # HACK: quick, untested, dirty hack for BPE tokenization, e.g.
            # This token will decompose. --> ĠThis Ġtoken Ġwill Ġdecom pose .
            if bpe and token.startswith('Ġ'):
                # remove token boundary
                token = token[1:]

            # remove leading spaces of current content
            content = content.lstrip()

            # if empty region, proceed to next region (keeping current token)
            if content == '':
                r_idx += 1
                r = self.regions[r_idx]
                content = r.content.lstrip()
            
            # if exact match with beginning of content, or unk
            elif content.startswith(token) or self.unks[t_idx] == 1:
                # add token to list of tokens for current region
                region_tokens[r.region_number].append(token)
                
                # remove token from content
                content = content[len(token):]

                # if end of content (removing spaces), and before last region
                if content.strip() == '' and r_idx < len(self.regions) - 1:
                    r_idx += 1
                    r = self.regions[r_idx]
                    content = r.content
                    
            # otherwise, move to next region and token
            else:
                r_idx += 1
                r = self.regions[r_idx]
                content = r.content
        return region_tokens

class Region:
    def __init__(self, region_number=None, content='', metric_value=None):
        utils.save_args(vars())
        self.token_surprisals = []

    def __repr__(self):
        s = 'Region(\n\t{}\n)'.format(str(vars(self)))
        return s

    def agg_surprisal(self, metric):
        self.surprisal = utils.METRICS[metric](self.token_surprisals)
        return self.surprisal