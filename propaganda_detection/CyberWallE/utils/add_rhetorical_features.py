"""
Parses the argument lexicon by Somasundaran, Ruppenhofer & Wiebe (2007):
http://people.cs.pitt.edu/~wiebe/pubs/papers/sigdial07.pdf
http://mpqa.cs.pitt.edu/lexicons/arg_lexicon/

Span identification task: Encodes whether a given token is contained in a
phrase that matches a rhetorical pattern.
Technique classification: Encodes whether a given fragment contains such a
rhetorically salient phrase.
"""
import re


path = '../data/arglex/'
strategies_5 = ['authority', 'doubt', 'emphasis', 'generalization', 'priority']
strategies_full = ['assessments', 'authority', 'causation', 'conditionals',
                   'contrast', 'difficulty', 'doubt', 'emphasis',
                   'generalization', 'inconsistency',
                   'inyourshoes', 'necessity', 'possibility', 'priority',
                   'rhetoricalquestion', 'structure', 'wants']
macros = ['modals', 'spoken', 'wordclasses', 'pronoun', 'intensifiers']
# macro -> list of expansions
expansions = dict()
# strategy -> list of regexes
regexes = dict()


def init(strategies, verbose=False):
    for macro in macros:
        with open(path + macro + '.tff') as f:
            for line in f:
                if line.startswith('#'):
                    # comment
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                fields = line.split('=')
                macro_word = fields[0]
                expansion_list = fields[1][1:-1]  # Strip away { and }
                # The lists use both ', ' and ',' as separators.
                expansion_list = expansion_list.replace(', ', ',')
                expansion_list = expansion_list.split(',')
                expansion_list = '(' + '|'.join(expansion_list) + ')'
                expansions[macro_word] = expansion_list

    if verbose:
        print('Macros and their expansions:')
        for m in expansions:
            print(m, expansions[m])
        print()

    for strategy in strategies:
        regexes[strategy] = []
        with open(path + strategy + '.tff') as f:
            for line in f:
                if line.startswith('#'):
                    # comment
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                for macro in expansions:
                    line = line.replace('(' + macro + ' )?',
                                        '(' + expansions[macro] + ' )?')
                    line = line.replace('(' + macro + ')', expansions[macro])
                line = line.replace('\\', '')
                regexes[strategy] += ['\\b' + line + '\\b']

    if verbose:
        print('Regexes for rhetorical strategies:')
        for s in regexes:
            print(s, regexes[s])
        print()


def parse_demo_file(filename):
    with open(filename, encoding='utf8') as f:
        for line in f:
            line = line.strip().lower()
            # Only for testing with data\arglex\patterntest.txt:
            line = line.replace('\\', '')
            print(line)
            for strategy in regexes:
                for regex in regexes[strategy]:
                    for match in re.finditer(regex, line):
                        print(strategy.upper(), '--', match.group(),
                              '--', match.span(), '--', regex)
            print()


def find_rhetorical_strategies(token_list, strategy):
    sentence = ' '.join(token_list)
    token_indices = set()
    if strategy is 'any':
        strats = [s for s in regexes]
    else:
        strats = [strategy]
    for strategy in strats:
        for regex in regexes[strategy]:
            for match in re.finditer(regex, sentence):
                # print(strategy.upper(), '--', match.group(),
                #       '--', match.span(), '--', regex)
                start_idx = match.span()[0]
                end_idx = match.span()[1]
                # idx in the token list
                token_indices.add(sentence[:start_idx].count(' '))
                token_indices.add(sentence[:end_idx].count(' '))
    return token_indices


def parse_input_file_si(infile, outfile, full=True, indiv_cols=False):
    """
    full: if True, use all strategies, if false, use the 5 most important
           strategies. Used for generating the preprocessing description.
           Should match the initialization
    indiv_cols: if True, each rhetorical strategy is represented by its own
                column. If False, matches for any strategy are represented
                in a single joint feature column.
    """
    with open(infile, encoding='utf8') as f_in:
        lines = f_in.readlines()
        lines.append('eof\teof\teof\teof\teof\teof\n')

    if indiv_cols:
        strategies = [s for s in regexes]
    else:
        strategies = ['any']

    with open(outfile, 'w', encoding='utf8') as f_out:
        rows = []
        tokens = []
        prev_article = ''
        first_line = True
        for line in lines:

            # Comments + header
            if line.startswith('#'):
                f_out.write(line)
                continue
            if first_line:
                f_out.write('# rhetorical_features: ArguingLexicon (')
                if full:
                    f_out.write('full, ')
                else:
                    f_out.write('5 main strategies, ')
                if indiv_cols:
                    f_out.write('individual feature columns')
                else:
                    f_out.write('joint feature column')
                f_out.write(')\n')
                first_line = False
                labels = line.strip().split('\t')
                try:
                    doc_idx = labels.index('document_id')
                except ValueError:
                    doc_idx = 0
                try:
                    word_idx = labels.index('token')
                except ValueError:
                    word_idx = 4
                for strategy in strategies:
                    labels.append('arglex_' + strategy)
                f_out.write('\t'.join(labels) + '\n')
                continue

            line = line[:-1]  # Remove \n
            fields = line.split('\t')
            article = fields[doc_idx]
            word = fields[word_idx]

            if article != prev_article:
                for strategy in strategies:
                    indices = find_rhetorical_strategies(tokens, strategy)
                    rows_new = []
                    for i, row in enumerate(rows):
                        if i in indices:
                            rows_new.append(row + '\t1')
                        else:
                            rows_new.append(row + '\t0')
                    rows = rows_new
                for row in rows:
                    f_out.write(row + '\n')
                tokens = []
                rows = []
            tokens.append(word)
            rows.append(line)
            prev_article = article


def annotate_tc(in_file):
    strategies = [s for s in regexes]
    strategies.sort()
    matched_strategies = {}

    with open(in_file, encoding='utf8') as f:
        lines = f.readlines()

    with open(in_file, 'w', encoding='utf8') as f:
        f.write(lines[0].strip() + '\t' + '\t'.join(strategies) + '\n')
        for line in lines[1:]:
            f.write(line.strip())
            text = line.split('\t')[4].strip().lower()
            for strategy in strategies:
                matched = 0
                for regex in regexes[strategy]:
                    for match in re.finditer(regex, text):
                        matched = 1
                        break
                f.write('\t' + str(matched))
                if matched:
                    try:
                        matched_strategies[strategy] += 1
                    except KeyError:
                        matched_strategies[strategy] = 1
            f.write('\n')
    for matched_strat in matched_strategies:
        strategies.remove(matched_strat)
    matched_strategies = sorted(matched_strategies.items(),
                                key=lambda s: s[1], reverse=True)
    for s in matched_strategies:
        print(s)
    print("Strategies without occurrences:", strategies)
    print()


if __name__ == "__main__":
    ### Task 1: Span identification
    init(strategies_full)
    parse_input_file_si('../data/train-improved-sentiwordnet.tsv',
                        '../data/train-improved-sentiwordnet-arguingfullindiv.tsv')
    parse_input_file_si('../data/dev-improved-sentiwordnet.tsv',
                        '../data/dev-improved-sentiwordnet-arguingfullindiv.tsv')
    parse_input_file_si('../data/test-improved-sentiwordnet.tsv',
                        '../data/test-improved-sentiwordnet-arguingfullindiv.tsv')

    ### Task 2: Technique identification
    init(strategies_full)
    # These strategies don't actually appear in the training data:
    regexes.pop('inyourshoes')
    regexes.pop('doubt')
    # These strategies barely appear in the training data (<10 occurrences):
    regexes.pop('difficulty')
    regexes.pop('conditionals')
    regexes.pop('assessments')
    regexes.pop('rhetoricalquestion')
    annotate_tc('../data/tc-train.tsv')
    annotate_tc('../data/tc-dev.tsv')
    annotate_tc('../data/tc-test.tsv')
