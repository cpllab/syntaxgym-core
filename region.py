import os
import utils
import subprocess
import tempfile

class Sentence:
    def __init__(self, condition_name='', regions=None, model=None):
        self.regions = [Region(**r) for r in regions]
        self.content = ' '.join(r.content for r in self.regions)
        self.tokens = self.tokenize(model=model)
        self.region2tokens = self.tokenize_regions()
        for i, r in enumerate(self.regions):
            r.tokens = self.region2tokens[r.region_number]
            self.regions[i] = r

    def tokenize(self, model=None):
        if model is None:
            return self.content.split(' ')
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))

            with tempfile.NamedTemporaryFile("w") as f:
                f.write(self.content)

                # feed temp file into tokenizer
                cmd = 'docker run --rm cpllab/language-models:%s tokenize /dev/stdin < %s' % (model, f.name)
                cmd = cmd.split()
                tokens = subprocess.run(cmd, stdout=subprocess.PIPE)

            tokens = tokens.stdout.decode('utf-8').strip().split(' ')
            return tokens

    def tokenize_regions(self):
        """
        Converts self.tokens (list of tokens) to dict of <region_number, token list> pairs.
        """
        # set current region
        cur_region_idx = 0
        r = self.regions[cur_region_idx]
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
                    # if end of content, and before last region
                    if content == '' and cur_region_idx < len(self.regions) - 1:
                        cur_region_idx += 1
                        r = self.regions[cur_region_idx]
                        content = r.content
                # otherwise, move to next region
                else:
                    cur_region_idx += 1
                    r = self.regions[cur_region_idx]
                    content = r.content
        return region_tokens

class Region:
    def __init__(self, region_number=None, content=''):
        utils.save_args(vars())
        self.token_surprisals = []

    def __repr__(self):
        s = '''Region(
    {}
)
    '''.format(str(vars(self)))
        return s

    def agg_surprisal(self, metric):
        self.surprisal = utils.METRICS[metric](self.token_surprisals)
        return self.surprisal
