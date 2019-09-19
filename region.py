import utils

class Sentence:
    def __init__(self, condition_name='', regions=None, tokens=[]):
        self.regions = [Region(**r) for r in regions]
        self.content = ' '.join(r.content for r in self.regions)
        self.tokens = tokens
        self.region2tokens = self.tokenize_regions()
        for i, r in enumerate(self.regions):
            r.tokens = self.region2tokens[r.region_number]
            self.regions[i] = r

    def tokenize_regions(self):
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
            # handle <eos> separately
            if token == '<eos>':
                region_tokens[r.region_number].append(token)
            else:
                # remove leading spaces of current content
                content = content.lstrip()
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
