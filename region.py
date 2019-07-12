import utils
import subprocess

class Region:
    def __init__(self, region_number=None, content='', tokenizer=None):
        utils.save_args(vars())
        if self.tokenizer is None:
            self.tokens = content.split(' ')
        else:
            self.tokens = self.tokenize(self.tokenizer)
        self.token_surprisals = []

    def __repr__(self):
        s = '''Region(
    {}
)
    '''.format(str(vars(self)))
        return s

    def tokenize(self, tokenizer):
        command = '%s %s' % (tokenizer, self.content)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output

    def agg_surprisal(self, metric):
        self.surprisal = utils.METRICS[metric](self.token_surprisals)
        return self.surprisal