import sys
import pickle
import subprocess
import nltk
nltk.download('punkt')
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
import warnings
warnings.filterwarnings(action='ignore')


PATH = '/jar'


def _format(sentence):
    path_to_jar = PATH + '/stanford-parser.jar'
    path_to_models_jar = PATH + '/stanford-parser-models.jar'
    dependency_parser = StanfordDependencyParser(
        path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    tokens = word_tokenize(sentence)
    result = dependency_parser.raw_parse(sentence)
    for dep in result:
        return (dep, tokens)


def parse_input_tree(tree):
    tree = tree.split('\n')
    tree_parsed = []
    for line in tree:
        tree_parsed.append(line.split('\t'))
    return tree_parsed


def add_tokens_to_sentence(parsed_dep_tree, tokens):
    tree = parsed_dep_tree[:]
    j = 0
    for i in range(len(parsed_dep_tree)):
        line = parsed_dep_tree[i]
        if int(line[0]) == (i + j + 1):
            continue
        else:
            if (i + j) < len(tree):
                tree.insert(i + j, [
                    str(i + j + 1), tokens[i + j], '_', 'PUNCT', 'PUNCT', '_',
                    '-10000', '_', '_', '_'
                ])
            else:
                tree.append([
                    str(i + j + 1), tokens[i + j], '_', 'PUNCT', 'PUNCT', '_',
                    '-10000', '_', '_', '_'
                ])
            j = j + 1
    return tree


def create_dict(tree):
    dict_list = dict()
    dep_tree = dict()
    dict_list[0] = {'index': 0, 'word': '///root', 'root': None}
    for line in tree:
        if len(line) < 2:
            continue
        d_temp = {
            'index': int(line[0]),
            'word': line[1],
            'rootword': line[2],
            'cpos': line[3],
            'pos': line[4],
            'tam': line[5],
            'root': int(line[6]),
            'rel': line[7],
            'rootrel': line[8],
            'other': line[9]
        }
        dict_list[d_temp['index']] = d_temp
        dep_tree[d_temp['index']] = d_temp['root']
    return dict_list, dep_tree


def return_root(tree):
    for line in tree:
        if tree[line]['root'] == 0:
            return line
    return False


def return_children(dep_tree, parent):
    children = [key for (key, value) in dep_tree.items() if value == parent]
    return children


def create_clause_sentence(clause_dict, dict_of_tree, newclause, output_file):
    xi = 1
    for index, newclause in sorted(clause_dict.items(), key=lambda hoc: hoc[1][-1]):
        if newclause == []:
            continue
        wordclause = dict_of_tree[newclause.pop(0)]['word']
        for ind in newclause:
            if ind not in dict_of_tree:
                continue
            if dict_of_tree[ind]['cpos'] == 'PUNCT':
                wordclause += dict_of_tree[ind]['word']
            else:
                wordclause += ' ' + dict_of_tree[ind]['word']
        output_file.write("Clause" + str(xi) + ": " + wordclause + "\n")
        xi += 1
    output_file.write("\n\n")
    output_file.close()


def get_index(dict1, tree, word_1, word_2):
    l = []
    for k, v in dict1.items():
        if v == word_1[0] and word_1 in tree[word_2]:
            l.append(k)
    if l == []:
        return 100000
    return l[0]


def get_sentence_dict():
    clause_dict = {}
    with open('./clause_output.txt', 'r') as out_f:
        for line in out_f.readlines():
            if line.split(':')[0] == 'Input Sentence':
                sentence = line.split(':')[1].split('\"')[0].strip()
            elif 'Clause' in line.split(':')[0]:
                key = line.split(':')[0].strip()[6]
                value = line.split(':')[1].strip()
                clause_dict[key] = value

    return sentence, clause_dict


def find_clause_breakpoints(clause_marked, tree, sentence_dict):
    clause_breakpoint = {}
    for k, v in clause_marked.items():
        word_1, relation, word_2 = v
        nearest_noun_index = 100000
        if word_2 not in tree:
            continue
        for w, crel in tree[word_2].items():
            if w[1][0] == 'N' or crel == 'nsubj' or crel == 'expl' or crel == 'advmod':
                temp = min(get_index(sentence_dict, tree, w, word_2),
                           get_index(sentence_dict, tree, w, word_2))
                if temp < nearest_noun_index:
                    nearest_noun_index = temp

        if nearest_noun_index == 100000:
            continue
        clause_breakpoint[v] = (nearest_noun_index,
                                sentence_dict[nearest_noun_index])
    return clause_breakpoint

def main(input_sentence):
    multi_clause_list = []
    verb_list = ['V']
    ignCount = 0
    dependency_tree, tokens = _format(input_sentence)
    dependency_tree = dependency_tree.to_conll(10)
    parsed_dep_tree = parse_input_tree(dependency_tree)
    if parsed_dep_tree[-1] == ['']:
        del parsed_dep_tree[-1]
    parsed_dep_tree = add_tokens_to_sentence(parsed_dep_tree, tokens)
    output_file = open('clause_output.txt', 'w')
    output_file.write("Input Sentence: " + input_sentence + '\n')
    multi_clause_list = [parsed_dep_tree]
    clause_relations = ['parataxis', 'ccomp', 'acl', 'acl:relcl', 'advcl', 'conj']
    for tree in multi_clause_list:
        dict_of_tree, dep_tree = create_dict(tree)
        root = return_root(dict_of_tree)
        coverage = dict()
        clause_dict = dict()
        queue = [0]
        clause = []
        head_of_clause = [key for (key, value) in dep_tree.items()
                          if value == 0][0]
        nextclause = []
        _itr = 0
        sentence = '"'
        sentence += dict_of_tree[1]['word']
        for index in range(2, len(dict_of_tree)):
            if index not in dict_of_tree:
                continue
            if dict_of_tree[index]['cpos'] == 'PUNCT':
                sentence += dict_of_tree[index]['word']
            else:
                sentence += ' ' + dict_of_tree[index]['word']
        while queue != [] or nextclause != []:
            if queue == []:
                queue = [nextclause[0]]
                newclause = []
                clause = sorted(clause)
                hoc = head_of_clause
                while hoc in clause:
                    if hoc == 0:
                        break
                    if dict_of_tree[hoc]['cpos'] == 'PUNCT':
                        break
                    newclause = [hoc] + newclause
                    coverage[hoc] = 1
                    hoc -= 1
                hoc = head_of_clause + 1
                while hoc in clause:
                    if dict_of_tree[hoc]['cpos'] == 'PUNCT':
                        break
                    newclause.append(hoc)
                    coverage[hoc] = 1
                    hoc += 1
                clause_dict[_itr] = newclause
                _itr += 1
                clause = []
                head_of_clause = nextclause.pop(0)
            parent = queue.pop(0)
            clause.append(parent)
            children = return_children(dep_tree, parent)
            newchildren = children
            for child in children:
                if dict_of_tree[child]['rel'] in clause_relations and dict_of_tree[
                        child]['pos'][0] == 'V':
                    nextclause.append(child)
                    newchildren.remove(child)
            queue = newchildren + queue
        newclause = []
        hoc = head_of_clause
        while hoc in clause:
            newclause = [hoc] + newclause
            coverage[hoc] = 1
            hoc -= 1
        hoc = head_of_clause + 1
        while hoc in clause:
            newclause.append(hoc)
            coverage[hoc] = 1
            hoc += 1
        clause = sorted(clause)
        clause_dict[_itr] = newclause
        _itr = 0
        clause = []
        xin = [
            hoc for hoc in range(1, len(dict_of_tree))
            if hoc not in coverage.keys()
        ]
        while xin != []:
            ind = xin.pop(0)
            for key, clause in sorted(clause_dict.items(),
                                      key=lambda hoc: hoc[1][-1]):
                if (ind + 1 in clause or ind - 1 in clause):
                    clause.append(ind)
                    clause_dict[key] = sorted(clause)
                    coverage[ind] = 1
                    break
            if ind not in coverage:
                xin = xin + [ind]
        create_clause_sentence(clause_dict, dict_of_tree, newclause, output_file)

    clause_dict = dict()
    sentence = str()
    clause_marked = {}
    tree = {}
    clause_relations = ['parataxis', 'ccomp', 'acl', 'acl:relcl', 'advcl', 'conj']
    sentence, clause_dict = get_sentence_dict()
    if len(clause_dict.keys())>1:
        sentence_dict = {k: v for k, v in enumerate(sentence.split())}
        parsed, tokens = _format(sentence)
        input_tree = parsed.to_conll(4)
        pos_tags = [
            line.split('\t')[1] for line in input_tree.split('\n')
            if line != '' and line.split('\t')[1] != 'POS'
        ]
        triples = parsed.triples()
        conll = parsed.to_conll(10)
        dot = parsed.to_dot()
        for p in triples:
            word_1, relation, word_2 = p
            if word_1 not in tree:
                tree[word_1] = {}
            if word_2 not in tree[word_1]:
                tree[word_1][word_2] = relation
            if relation in clause_relations:
                clause_marked[relation] = p
        clause_breakpoint = find_clause_breakpoints(clause_marked, tree, sentence_dict)
        break_point = [v[0] for v in clause_breakpoint.values()]
        break_point = sorted(break_point)
        if len(break_point) != 0 :
            xc = 1
            Sentences = {}
            for k, v in sorted(sentence_dict.items(), key=lambda x: x[0]):
                if k == break_point[0]:
                    xc += 1
                if pos_tags[k] == 'CC' or pos_tags[k] == 'WRB' or pos_tags[k] == 'WDT':
                    continue
                stored_p = ()
                stored_l = []
                for key in clause_breakpoint:
                    if v == key[2][0] and key[1] == 'acl:relcl':
                        stored_p = key[0]
                if stored_p != ():
                    for asd in tree[stored_p]:
                        if asd[0] != v:
                            stored_l.append(asd[0])
                    stored_l.append(stored_p[0])
                    v += ' ' + ' '.join(stored_l)
                cn = 'Clause' + str(xc)
                if cn not in Sentences:
                    Sentences[cn] = ''
                    v = v.capitalize()
                Sentences[cn] += v + ' '
            tmp_sent = []
            for k, v in Sentences.items():
                tmp_sent.append(v.strip())
        else:
            tmp_sent=[input_sentence]
    else:
      tmp_sent=[input_sentence]

    return tmp_sent
