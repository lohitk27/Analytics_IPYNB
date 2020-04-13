
# coding: utf-8

# In[1]:

import pandas as pd
import scipy as sp
import numpy as np


# In[2]:

philips = pd.ExcelFile("D:/Philips_Excel/USA.xlsx")


# In[3]:

philips.sheet_names


# In[4]:

pdata=philips.parse('Sheet1')


# In[5]:

pdata


# In[6]:

from nltk.corpus import brown


# In[7]:

from nltk import sent_tokenize, word_tokenize, pos_tag


# In[10]:

##### Create an Empty array for considering list all objects into an array
import numpy as np
descrs=[]
for i in range(0,11):
    descr =  pdata['CustomerComplaint'][i]
    descrs.append(descr)


print descrs


# In[11]:

data1= ''.join(descrs)  ##### Converting an array to String format 


# In[12]:

sents = sent_tokenize(data1)  #####Sent_tokenize


# In[13]:

print len(data1)  ##### Check Length of the data if we have any doubt with the string format & We Can Lookup into each string also.


# In[14]:

print data1[21000:21877]


# In[15]:

tokens = word_tokenize(data1)  ##### Word tokenize
print tokens


# In[16]:

pos_tokens = pos_tag(data1)
print pos_tokens


# In[17]:

##### We Can get what tag it is from the above cell NLTK has provided an Documentation for Each of Parts of Speech 
import nltk
nltk.help.upenn_tagset('RB')


# In[18]:

from collections import Counter   ### We Need to import Counter for the counting of Text Tokens
totalWords9 = Counter(data1)
print totalWords9


# In[19]:

sorted(set(data1))


# In[20]:

print len(set(data1))


# In[21]:

cus_tokens = data1.split()
print cus_tokens


# In[22]:

from collections import Counter   ### We Need to import Counter for the counting of Text Tokens
totalWords9 = Counter(cus_tokens)
print totalWords9


# In[23]:

low_Cust_tokens = data1.lower() ### Lower is a type to bring all the words in lower format ease to understand the count of words
print low_Cust_tokens


# In[24]:

import re
processedText = re.sub(r'[^ a-z 0-9  \s]', '', low_Cust_tokens) 
print processedText


# In[25]:

from gensim import corpora


# In[26]:

def __init__(self, documents=processedText):
    self.token2id = {} # token -> tokenId
    self.id2token = {} # reverse mapping for token2id; only formed on request, to save memory
    self.dfs = {} # document frequencies: tokenId -> in how many documents this token appeared

    self.num_docs = 0 # number of documents processed
    self.num_pos = 0 # total number of corpus positions
    self.num_nnz = 0 # total number of non-zeroes in the BOW matrix

    if documents is not None:
        self.add_documents(documents)


# In[28]:

def termdocumentmatrix_example():
    # Create some very short sample documents
    doc1 = 'John and Bob are brothers.'
    doc2 = 'John went to the store. The store was closed.'
    doc3 = 'Bob went to the store too.'
    # Initialize class to create term-document matrix
    tdm = textmining.TermDocumentMatrix()
    # Add the documents
    tdm.add_doc(doc1)
    tdm.add_doc(doc2)
    tdm.add_doc(doc3)
    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
    tdm.write_csv('D:/Philips/matrix.csv', cutoff=1)
    # Instead of writing out the matrix you can also access its rows directly.
    # Let's print them to the screen.
    row=[]
    for row in tdm.rows(cutoff=1):
        print row
        



# In[29]:

from __future__ import print_function
from nltk.stem import *


# In[30]:

from nltk.stem.porter import *
stemmer = PorterStemmer()


# In[31]:

plurals = processedText


# In[32]:

singles = [stemmer.stem(plural) for plural in plurals]


# In[33]:

print(' '.join(singles))


# In[34]:

from nltk.stem.snowball import SnowballStemmer
print(" ".join(SnowballStemmer.languages))


# In[35]:

stemmer = SnowballStemmer("english")


# In[42]:

print(stemmer.stem("running"))


# In[43]:

from sklearn.feature_extraction.text import CountVectorizer


# In[44]:

countvec = CountVectorizer()


# In[48]:

from gensim import corpora, models, similarities


# In[50]:

tfidf = models.TfidfModel(corpus)


# In[51]:

corpus_tfidf = tfidf[processedText]


# In[52]:

v1 = CountVectorizer(max_df=0.5)


# In[55]:

Text=iter(low_Cust_tokens[:-1])
counts_train = v1.fit_transform(Text)


# In[56]:

def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [low_Cust_tokens for low_Cust_tokens in tokens if low_Cust_tokens not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens.append(" ".join(original_tokens[i: i + n]))

        return tokens


# In[62]:

def _char_ngrams(self, processedText):
        """Tokenize text_document into a sequence of character n-grams"""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", processedText)

        text_len = len(text_document)
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in xrange(min_n, min(max_n + 1, text_len + 1)):
            for i in xrange(text_len - n + 1):
                ngrams.append(text_document[i: i + n])
        return ngrams


# In[65]:

def _char_wb_ngrams(self, processedText):
        """Whitespace sensitive char-n-gram tokenization.
        Tokenize text_document into a sequence of character n-grams
        excluding any whitespace (operating only inside word boundaries)"""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", processedText)

        min_n, max_n = self.ngram_range
        ngrams = []
        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in xrange(min_n, max_n + 1):
                offset = 0
                ngrams.append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams.append(w[offset:offset + n])
                if offset == 0:   # count a short word (w_len < n) only once
                    break
        return ngrams


# In[66]:

def fit(self, processedText, y=None):
        """Learn vocabulary and idf from training set.
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        self : TfidfVectorizer
        """
        X = super(TfidfVectorizer, self).fit_transform(processedText)
        self._tfidf.fit(X)
        return self


# In[67]:

def fit_transform(self, processedText, y=None):
        """Learn vocabulary and idf, return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        X = super(TfidfVectorizer, self).fit_transform(processedText)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)


# In[68]:

def transform(self, processedText, copy=True):
        """Transform documents to document-term matrix.
        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = super(TfidfVectorizer, self).transform(processedText)
        return self._tfidf.transform(X, copy=False)


# In[69]:

from sklearn.feature_extraction.text import CountVectorizer


# In[70]:

vectorizer = CountVectorizer(min_df=1)


# In[71]:

vectorizer


# In[72]:

corpus = [processedText]


# In[73]:

X = vectorizer.fit_transform(corpus)


# In[74]:

X


# In[75]:

analyze = vectorizer.build_analyzer()


# In[90]:

vectorizer.get_feature_names() == ([processedText])


# In[77]:

X.toarray()


# In[92]:

X.toarray(['problem '])


# In[87]:

vectorizer.vocabulary_.get('by')


# In[91]:

vectorizer.transform(['problem reported by customererror']).toarray()


# In[ ]:


###############################################################ALL Values.py######################################################

from __future__ import print_function
import cv2
import numpy as np
# image = cv2.imread('15.png')
# #image = cv2.medianBlur(image,3)
# #cv2.imshow('14.png',image)
# #cv2.waitKey(0)
# gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #gs_image = cv2.fastNlMeansDenoising(gs_img, None, 65, 5, 21)

# rows, cols = gs_image.shape
# gs_image_padded = gs_image[200:rows - 200, 200:]
# gs_image_padded = 255 - gs_image_padded
# print(gs_image_padded.shape, rows, cols)
# gs_image_padded[gs_image_padded < 128] = 0 
# gs_image_padded[gs_image_padded > 128] = 255

image = cv2.imread('15.png')
#image = cv2.medianBlur(image,3)
#cv2.imshow('14.png',image)
#cv2.waitKey(0)
gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gs_image = cv2.fastNlMeansDenoising(gs_img, None, 65, 5, 21)

rows, cols = gs_image.shape

# Assuming that print stray marks occur in the page boundaries.
gs_image_padded = gs_image[0:rows - 0, 0:]

# Inverting the colors
gs_image_padded = 255 - gs_image_padded
gs_image_padded[gs_image_padded < 128] = 0 
gs_image_padded[gs_image_padded > 128] = 255
#print (gs_image_padded.shape)
def lineExtract(image):
		def getHorizontalLimits(img):
			INFINITY = 100000
			mn, mx = INFINITY, 0
			for r in img:
				non_zero = np.where(r == 255)
				mn = min(mn, np.min(non_zero))
				mx =  max(mx, np.max(non_zero))
			return mn, mx

		image[image < 128] = 0
		image[image > 128] = 255	
		rows, cols = image.shape
		
		lines = [] #Stores the lines
		line_of_text = []
		start_row=[]
		end_row=[]
		line_no=[]
		start_col=[]
		end_col=[]

		row_index = 0
		line_no = 1
		tracking = False
		this_line_start_index = 0

		for row in image:
			#Checking if atleast one nonzero pixel exist in row
			if not tracking:
				this_line_start_index = row_index

			if np.count_nonzero(row) > 0:
				line_of_text.append(row)
				tracking = True

			elif np.array(line_of_text).shape[0] > 25:
				start_col, end_col = getHorizontalLimits(np.array(line_of_text))
				lines.append({"start_row":this_line_start_index,
			    	"end_row":row_index,
					"line_no":line_no,
					"start_col":start_col,
					"end_col":end_col,
					"pixels":np.array(line_of_text)
					})
				line_of_text = []
				line_no+=1
				tracking = False

				row_index += 1
		return lines
lines = lineExtract(gs_image_padded)
for line in lines:
	#first_row.append(this_line_start_index)

	print("starting row: ", line["start_row"])
	print("ending row: ", line["end_row"])
	print("starting col: ", line["start_col"])
	print("ending col: ", line["end_col"])
	print("Line Number: ", line["line_no"])
	print("pixels: ", line["pixels"])
    
    
####################################################################################################################################################
 ##################### AWS Database Mangement#################################################
 #!/pkg/ldc/bin/python2.1
#-----------------------------------------------------------------------------
# Name:        AWNDatabaseManagement.py
# Purpose:
#
# Author:      Horacio
#
# Created:     2008/06/10
# Load AWN database and provide some simple access functions
#-----------------------------------------------------------------------------

import xml.parsers.expat
import re
from string import *
import sys

class ITEM:
    "item tuple"
    def __init__(self,itemid,offset,name,type,pos):
        self.__init_vars(itemid,offset,name,type,pos)
    def __init_vars(self,itemid,offset,name,type,pos):
        self._content=[offset,name,type,pos]
        self._itemid=itemid
        self._links_out=[]
        self._links_in=[]
    def get_itemid(self):
        return self._itemid
    def get_offset(self):
        return self._content[0]
    def get_name(self):
        return self._content[1]
    def get_type(self):
        return self._content[2]
    def get_pos(self):
        return self._content[3]
    def put_link(self,itemid1,itemid2,type):
        if self._itemid == itemid1:
            self._links_out.append([type,itemid2])
        elif self._itemid == itemid2:
            self._links_in.append([type,itemid1])
##get_links:
##    input:
##        direcc: direction of the link
##            'in' or 'out'
##        type: type of the link (e.g. 'has_hyponym')
##            default 'all'
##    output:
##        list of tuples:
##            each tuple contains:
##                type of link
##                the other item of the link
    def get_links(self,direcc,type='all'):
        if direcc == 'in':
            return filter(lambda(x):(x[0]==type) or (type=='all'),self._links_in)
        else:
            return filter(lambda(x):(x[0]==type) or (type=='all'),self._links_out)
    def describe(self):
        print 'itemid ', self._itemid
        print 'offset ', self._content[0]
        print 'name ', self._content[1]
        print 'type ', self._content[2]
        print 'pos ', self._content[3]
        print 'input links ',self._links_in
        print 'output links ',self._links_out

class WORD:
    "word tuple"
    def __init__(self,wordid,value,synsetid):
        self.__init_vars(wordid,value,synsetid)
    def __init_vars(self,wordid,value,synsetid):
        self._value=value
        self._wordid=wordid
        self._synsets=[synsetid]
        self._forms=[]
    def get_roots(self):
        return filter(lambda(x):(x[0]=='root'),self._forms)
    def get_forms(self,type='all'):
        return filter(lambda(x):(x[0]==type) or (type=='all'),self._forms)
    def put_form(self,form,type):
        self._forms.append((type,form))
    def put_synset(self,synsetid):
        self._forms.append(synsetid)
    def describe(self):
        print 'wordid ', self._wordid
        print 'value ', self._value
        print 'synsets ', self._synsets
        print 'forms ', self._forms

class FORM:
    "form tuple"
    def __init__(self,form,wordid,type):
        self.__init_vars(form,wordid,type)
    def __init_vars(self,form,wordid,type):
        self._form=form
        self._words=[(type,wordid)]
    def put_word(self,wordid,type):
        self._words.append((type,wordid))
    def get_words(self,type='all'):
        return filter(lambda(x):(x[0]==type) or (type=='all'),self._words)
    def describe(self):
        print 'form ', self._form
        print 'words ', self._words

        

class WN:
    "representation of a WN"
    def __init__(self):
        self.__init_vars()
    
    def __init_vars(self):
        self._source_counts = {
            'item':0,
            'link':0,
            'word':0,
            'form':0,
            'verbFrame':0,
            'authorship':0,
            'all':0}
        self._items = {}
        self._words = {}
        self._forms = {}
        self._index_w = {}
        self._index_f = {}

##summary:
##    short description of the content of WN

    def summary(self):
        for i in self._source_counts.keys():
            print i+'\t'+str(self._source_counts[i])
  
    def update_item(self,itemid,offset,name,type,pos):
        if self._items.has_key(itemid):
            print 'itemid '+itemid+' duplicated, ignored'
        else:
            self._items[itemid]=ITEM(itemid,offset,name,type,pos)
        
    def update_link(self,itemid1,itemid2,type):
        if not self._items.has_key(itemid1):
            print 'itemid1 '+itemid1+' not present, ignored'
        elif not self._items.has_key(itemid2):
            print 'itemid2 '+itemid2+' not present, ignored'
        else:
            self._items[itemid1].put_link(itemid1,itemid2,type)
            self._items[itemid2].put_link(itemid1,itemid2,type)

    def update_word(self, wordid, value, synsetid):
        if not self._items.has_key(synsetid):
           print 'synsetid '+synsetid+' not present, ignored'
        else:
            if self._words.has_key(wordid):
                self._words[wordid].put_synset(synsetid)
            else:
                self._words[wordid] =WORD(wordid, value, synsetid)
            
    def update_form(self, value, wordid, type):
        if not self._words.has_key(wordid):
           print 'wordid '+wordid+' not present, ignored'
        else:
            self._words[wordid].put_form(value, type)
            if self._forms.has_key(value):
                self._forms[value].put_word(wordid, type)
            else:
                self._forms[value]=FORM(value,wordid, type)

    def compute_index_w(self):
        for i in self._words.keys():
            w=self._words[i]
            if self._index_w.has_key(w._value):
                self._index_w[w._value].append(w._wordid)
            else:
                self._index_w[w._value]=[w._wordid]

    def compute_index_f(self):
        for i in self._forms.keys():
            f=self._forms[i]
            if self._index_f.has_key(f._form):
                self._index_f[f._form].append(f.get_words())
            else:
                self._index_f[f._form]=f.get_words()

    def count_words(self):
        return len(self._words.keys())
    
    def count_forms(self):
        return len(self._forms.keys())

    def count_synsets(self):
        return len(filter(lambda(x):self._items[x].get_type()=='synset',self._items.keys()))

##get_words:
##    input:
##        simple: True or False
##    output:
##        if simple:
##            list of words
##        else:
##            list of pairs <word, wordid>
            
    def get_words(self,simple=False):
        if simple:
            return self._index_w.keys()
        else:
            l=[]
            for i in self._index_w.keys():
                l.append((i,self._index_w[i]))
            return l

##get_forms:
##    input:
##        simple: True or False
##    output:
##        if simple:
##            list of forms
##        else:
##            list of pairs <form, list of pairs <type, wordid>>
    
    def get_forms(self,simple=False):
        if simple:
            return self._index_f.keys()
        else:
            l=[]
            for i in self._index_f.keys():
                l.append((i,self._index_f[i]))
            return l

##get_wordids_from_word:
##    input: a word
##    output: list of wordid

    def get_wordids_from_word(self,word):
        if self._index_w.has_key(word):
            return self._index_w[word]
        else:
            return None

##get_forms_from_word:
##    input: a word
##    output: list of forms

    def get_forms_from_word(self,word):
        wis=self.get_wordids_from_word(word)
        if wis:
            forms=set()
            for i in wis:
                f=self._words[i].get_forms()
                if f:
                    forms.update(f)
            return list(forms)
        else:
            return None

##get_roots_from_word:
##    input: a word
##    output: list of roots

    def get_roots_from_word(self,word):
        wis=self.get_wordids_from_word(word)
        if wis:
            forms=set()
            for i in wis:
                f=self._words[i].get_roots()
                if f:
                    forms.update(f)
            return map(lambda(x):x[1],list(forms))
        else:
            return None

##get_synsetids_from_word:
##    input: a word
##    output: list of synsetids

    def get_synsetids_from_word(self,word):
        wis=self.get_wordids_from_word(word)
        if wis:
            synsets=set()
            for i in wis:
                synset=self._words[i]._synsets
                if synset:
                    synsets.update(synset)
            return list(synset)
        else:
            return None

##get_synsets_from_word:
##    input: a word
##    output: list of synsets


    def get_synsets_from_word(self,word):
        sids=self.get_synsetids_from_word(word)
        if sids:
            return map(lambda(x):(self._items[x].get_pos(),self._items[x].get_offset()),sids)
        else:
            return None

    def get_form(self,form):
        if self._index_f.has_key(form):
            return self._index_f[form]
        else:
            return None
    

    def count_forms(self):
        return len(self._forms.keys())

    def count_synsets(self):
        return len(filter(lambda(x):self._items[x].get_type()=='synset',self._items.keys()))


def processCmdlineOpts(cmdOpts):
    """ Process command line options; return a hash that can be passed
    to the application. """
    opts = {}
    for i in range(1,len(cmdOpts)):
        if re.match('-i', cmdOpts[i]):
            opts['i'] = cmdOpts[i+1]
    if not opts.has_key('i'):
        opts['i']='E:/usuaris/horacio/arabicWN/AWNdatabase/upc_db.xml'
    return opts


def start_element(name, attrs):
    global wn
    wn._source_counts['all']+=1
    for i in wn._source_counts.keys():
        if name == i:
            wn._source_counts[i]+=1
            break
    if name == 'item':
        wn.update_item(
            attrs['itemid'],
            attrs['offset'],
            attrs['name'],
            attrs['type'],
            attrs['POS'])
    elif name == 'link':
        wn.update_link(
            attrs['link1'],
            attrs['link2'],
            attrs['type'])
    elif name == 'word':
        wn.update_word(
            attrs['wordid'],
            attrs['value'],
            attrs['synsetid'])
    elif name == 'form':
        wn.update_form(
            attrs['value'],
            attrs['wordid'],
            attrs['type'])
    

def end_element(name):
    pass

def char_data(data):
    pass

def _encode(data):
    return data.encode('utf8')


def loadAWNfile(ent):
    global wn
    print 'processing file ', ent
    try:
        ent = open(ent)
        print ent
        wn=WN()
        p.ParseFile(ent)
    except IOError:
        print 'file ',ent,' not correct'

        
def test():
    "tests some functions"
    a=wn.get_words(True)
    print  'length of a: ', len(a)
    print 'a[0]: ', a[0]
    b=wn.get_forms(False)
    print  'length of b: ', len(a)
    print 'b[0]: ', b[0][0]
    for i in b[0][1]:
        print '\t',i[0],i[1]
    c=wn._items.keys()
    print  'length of c: ', len(c)
    print 'c[0]: ', c[0]
    wn._items[c[0]].describe()
    d=wn._words.keys()
    print  'length of d: ', len(d)
    print 'd[0]: ', d[0]
    wn._words[d[0]].describe()


opts = processCmdlineOpts(sys.argv)
p = xml.parsers.expat.ParserCreate()
p.StartElementHandler = start_element
p.EndElementHandler = end_element
p.CharacterDataHandler = char_data
loadAWNfile(opts['i'])
wn.compute_index_w()
wn.compute_index_f()

########################################################################################################################################################

# coding: utf-8

# In[73]:


text = "well name sai depth: 200 mRT well name:krishna depth (ft) 6545 depth of the well is 500ft Well No. 2A sdlkf "
text1 ="Total Depth (ft) (PBTD) 6545ft"
#text = "Subject ; Asuhaltene Studv' well: Bosi2 140 Sand depth: 2564.1 mRT (2636.7 mTVDSS) our le: ACL 200219410"
text = text.replace(":"," ")
text  = text.lower()


# In[114]:


text1


# In[1]:


import numpy as np
import nltk
SOME_FIXED_SEED = 42
np.random.seed(SOME_FIXED_SEED)


# In[2]:


def get_substring_indices(text1, q):
    result = [i for i in range(len(text1)) if text1.startswith(q, i)]
    k = []
    #print(result)
    for i in result:
        if i==0 or i==len(text1)-len(q):
            k.append(i)
        else:
            if text1[i-1]== ' ' and text1[i+len(q)]==' ':
                k.append(i)
    return k


# In[456]:


c = get_substring_indices('sai well name : fsdlkfj','well name')
print (c)


# In[3]:


import pandas as pd


# In[30]:


df = pd.read_csv("E:/Geo//test.csv")


# In[31]:


df['key'] ="NONE"
df['value']="NONE"


# In[7]:


df


# In[390]:


s


# In[ ]:





# In[432]:


g =' welln n 0. 118-cvx-2x&2xst1 well'
#TODO
get_substring_indices('o f do sphaerozdinella dehiscens (n l 9-n23) at 618m', 'saik')


# In[392]:


for i,j in enumerate(df['ocr_output']):
    df.loc[i,'key'] = j


# In[76]:


key = []
value=[]
ff = []
pp=['sai','depth']
for i in pp:
        #print(q)
    ind = get_substring_indices(text, i)
    new_list = [x+len(i) for x in ind]
        #if (i == 'depth'):
        #    for j in new_list:
        #        unit,val = depth(text[j:])
        #        print (val,unit)
    ff.append(new_list)
    if len(new_list) != 0:
        for j,e in enumerate(new_list):
            key.append(i)
            if(j == len(new_list)-1):
                value.append(text[e:])
            else:
                    
                value.append(text[e:new_list[j+1]-len(i)])
print(ff)
#print(value)


# In[82]:





# In[84]:





# In[80]:


ll


# In[18]:


def depth(val):
    #print(val)
    nouns_list = ['NN','NNP','$','CD','RB']
    units = ['ft','mrt','(ft)','m']
    val = val.split()
    #print(val)
    tagged = nltk.pos_tag(val)
    #print (tagged)
    tagged_dict = dict((x, y) for x, y in tagged)
    #print(tagged_dict)
    found = 0
    unit = 'Depth unit not found'
    previ = 'No depth value found'
    for i in val:
        if i in units:
            unit = i
        if found == 1:
            return previ,unit
        if tagged_dict[i] in nouns_list:
            if tagged_dict[i]=="CD":
                found = 1
                previ = i
                if len(val)==1:
                    return i,'No depth value found'
        else:
            return unit,'No depth value found'


# In[110]:


x,y= depth('792m)')
print(x,y)


# In[32]:


s = ['well','depth','name','interval','top']
#key =[]
#value =[]
for p,q in enumerate(df['ocr_output']):
    q = str(q)
    q  = q.lower()
    q = q.replace(':'," ")
    key = []
    value=[]
    for i in s:
        #print(q)
        ind = get_substring_indices(q, i)
        new_list = [x+len(i) for x in ind]
        #if (i == 'depth'):
        #    for j in new_list:
        #        unit,val = depth(text[j:])
        #        print (val,unit)
    
        if len(new_list) != 0:
            for j,e in enumerate(new_list):
                key.append(i)
                if(j == len(new_list)-1):
                    value.append(q[e:])
                else:
                    
                    value.append(q[e:new_list[j+1]-len(i)])
                #else:
                #    value.append(q[e:])
        #if i=="name" and len(new_list)!=0:
        #    print (new_list)
        #    print(value)
    aa = "|".join(key)
    bb = "|".join(value)
    #print(key)
    #print(value)
    df.loc[p,'key'] = aa
    df.loc[p,'value'] = bb


# In[35]:


df.to_csv("E:/Geo/test_out.csv")


# In[33]:


key_df=df[df['key']!='']


# In[34]:


key_df


# In[36]:


for i,j in enumerate(key_df['key']):
    splitted= j.split('|')
    splitted_value = key_df.iloc[i]['value'].split('|')
    for bb,zz in enumerate(splitted):
        if zz == 'depth':
            x,y = depth(splitted_value[bb])
            #print(splitted_value[bb])
            print(x,y)
            #p = key_df.iloc[i]['value']
            #q = key_df.iloc[i]['ocr_output']
            #print(splitted_value[bb])
            #print(q)
            #print(nltk.pos_tag(splitted_value[bb].split()))
            #print(nltk.pos_tag(q.lower().split()))
            print("$$$$$$$$$$$$$$$$$$$$$$$$")
        


# In[26]:


ss = ' 1400-4290m;'
nltk.pos_tag


# In[292]:


nltk.pos_tag(value[0].split())


# In[41]:


text1 = text1.lower()
tex = text1.split()


# In[36]:


import nltk


# In[55]:


val = ' 200 mrt well name:krishna'
val = val.split()


# In[57]:


tagged = nltk.pos_tag(val)


# In[66]:


tagged


# In[67]:


tagged_dict = dict((x, y) for x, y in tagged)


# In[134]:


text2 = ' (ft) 6545ft'
text2= text2.lower()
text2 = text2.split()


# In[135]:


nltk.pos_tag(text2)


# In[70]:


for i in val:
    if tagged_dict[i]=="CD":
        print (i)
    

#####################################################################################################################################################
# -*- coding: utf-8 JOHN DEERE -*-
"""
Created on Tue Jul 11 18:18:15 2017

@author: 20130407
"""


import os
os.chdir("D:/John Deere/dummy")



import pandas as pd
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


Part = pd.read_csv("Parts_Final.csv", header=None)
Tech = pd.read_csv("Tech_Final.csv", header=None, encoding = "ISO-8859-1")

'''
Part = Part.values.tolist()
Tech = Tech.values.tolist()

Results = pd.DataFrame({'Part': Part,'Tech': Tech})
Results['Similarity'] = ''
for row in list(range(0,len(Part))):
    Results['Similarity'][row] = cosine_sim(Part[row][0], Tech[row][0])
''' 

Results = pd.DataFrame(index=list(range(0,len(Part) * len(Tech))))
Results['Part'] = ''
Results['Tech'] = ''

for Partrow in list(range(0,len(Part))):
    for Techrow in list(range(0,len(Tech))):
        Results['Part'][Techrow+(len(Part)*Partrow)] = Part.iloc[Partrow][0]
        Results['Tech'][Techrow+(len(Part)*Partrow)] = Tech.iloc[Techrow][0]

Results['Similarity'] = ''
for Partrow in list(range(0,len(Part))):
    for Techrow in list(range(0,len(Tech))):
        Results['Similarity'][Techrow+(len(Part)*Partrow)] = cosine_sim(Results.iloc[Techrow+(len(Part)*Partrow)]['Part'], Results.iloc[Techrow+(len(Part)*Partrow)]['Tech'])


Results.to_csv('Results.csv')


abbrevations = {'HOF': ['Hydraulic Oil Filter'], 
'FH': ['Front Hitch'],
'HS': ['Hydraulic System'],
'HP': 'Hydraulic Pump',
'PFC': 'Pressure Flow Compensated'}

domain_synonyms_dict = {'Oil': ['fuel'], 
'pressure': ['stress'],
'vibration': ['shake'],
'measure': ['calculate']}

domain_taxonomy_dict = {'sports': ['baseball','cricket'], 
'filter': ['air filter', 'fuel filter', 'enfine oil filter'],
'Tractors': ['5C Series', '5D Series','5E Series']}












































############### Old Code ###################

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 

def fn_tdm_df(docs, xColNames = None, **kwargs):
    ''' create a term document matrix as pandas DataFrame
    with **kwargs you can pass arguments of CountVectorizer
    if xColNames is given the dataframe gets columns Names'''

    #initialize the  vectorizer
    vectorizer = CountVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)
    #create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index = vectorizer.get_feature_names())
    if xColNames is not None:
        df.columns = xColNames

    return df

DIR = 'D:/John Deere/dummy/Tech'

def fn_CorpusFromDIR(xDIR):
    ''' functions to create corpus from a Directories
    Input: Directory
    Output: A dictionary with 
             Names of files ['ColNames']
             the text in corpus ['docs']'''
    import os
    Res = dict(docs = [open(os.path.join(xDIR,f)).read() for f in os.listdir(xDIR)],
               ColNames = map(lambda x: 'P_' + x[0:30], os.listdir(xDIR)))
    return Res

d1 = fn_tdm_df(docs = fn_CorpusFromDIR(DIR)['docs'], 
               xColNames = fn_CorpusFromDIR(DIR)['ColNames'],
                                           stop_words=None)  


DIR = 'D:/John Deere/dummy/Part'

def fn_CorpusFromDIR(xDIR):
    ''' functions to create corpus from a Directories
    Input: Directory
    Output: A dictionary with 
             Names of files ['ColNames']
             the text in corpus ['docs']'''
    import os
    Res = dict(docs = [open(os.path.join(xDIR,f)).read() for f in os.listdir(xDIR)],
               ColNames = map(lambda x: 'P_' + x[0:30], os.listdir(xDIR)))
    return Res

d2 = fn_tdm_df(docs = fn_CorpusFromDIR(DIR)['docs'], 
               xColNames = fn_CorpusFromDIR(DIR)['ColNames'],
                                           stop_words=None) 



d1.to_csv('techdtm.csv')
d2.to_csv('partdtm.csv')














'''
import pandas as pd
Part = pd.read_csv("Parts.csv")
Tech = pd.read_csv("Tech1.csv", encoding = "ISO-8859-1")

import nltk
import textmining
import stemmer

def termdocumentmatrix_example():
    # Initialize class to create term-document matrix
    tdm = textmining.TermDocumentMatrix()
    # Add the documents
    tdm.add_doc(Part)
    tdm.add_doc(Tech)
    tdm.write_csv('matrix.csv', cutoff=2)
    # Instead of writing out the matrix you can also access its rows directly.
    # Let's print them to the screen.
    for row in tdm.rows(cutoff=2):
            print (row)

termdocumentmatrix_example()







from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import numpy
import gensim
import pandas as pd

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health." 

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)


# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

my_df = pd.DataFrame(corpus)

'''
########################################################################################################################################################

from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f1: {}".format(f1_score(true_value, pred)))


# our classifier to use
classifier = RandomForestClassifier

data = fetch_datasets()['wine_quality']

# splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=2)


# build normal model
pipeline = make_pipeline(classifier(random_state=42))
model = pipeline.fit(X_train, y_train)
prediction = model.predict(X_test)

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), classifier(random_state=42))
smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_test)

# build model with undersampling
nearmiss_pipeline = make_pipeline_imb(NearMiss(random_state=42), classifier(random_state=42))
nearmiss_model = nearmiss_pipeline.fit(X_train, y_train)
nearmiss_prediction = nearmiss_model.predict(X_test)



# print information about both models
print()
print("normal data distribution: {}".format(Counter(data['target'])))
X_smote, y_smote = SMOTE().fit_sample(data['data'], data['target'])
print("SMOTE data distribution: {}".format(Counter(y_smote)))
X_nearmiss, y_nearmiss = NearMiss().fit_sample(data['data'], data['target'])
print("NearMiss data distribution: {}".format(Counter(y_nearmiss)))

# classification report
print(classification_report(y_test, prediction))
print(classification_report_imbalanced(y_test, smote_prediction))

print()
print('normal Pipeline Score {}'.format(pipeline.score(X_test, y_test)))
print('SMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))
print('NearMiss Pipeline Score {}'.format(nearmiss_pipeline.score(X_test, y_test)))


print()
print_results("normal classification", y_test, prediction)
print()
print_results("SMOTE classification", y_test, smote_prediction)
print()
print_results("NearMiss classification", y_test, nearmiss_prediction)











from sklearn.model_selection import KFold

# cross validation done right
kf = KFold(n_splits=5, random_state=42)
accuracy = []
precision = []
recall = []
f1 = []
auc = []
for train, test in kf.split(X_train, y_train):
    pipeline = make_pipeline_imb(SMOTE(), classifier(random_state=42))
    model = pipeline.fit(X_train[train], y_train[train])
    prediction = model.predict(X_train[test])

    accuracy.append(pipeline.score(X_train[test], y_train[test]))
    precision.append(precision_score(y_train[test], prediction))
    recall.append(recall_score(y_train[test], prediction))
    f1.append(f1_score(y_train[test], prediction))
    auc.append(roc_auc_score(y_train[test], prediction))

print()
print("done right mean of scores 5-fold:")
print("accuracy: {}".format(np.mean(accuracy)))
print("precision: {}".format(np.mean(precision)))
print("recall: {}".format(np.mean(recall)))
print("f1: {}".format(np.mean(f1)))
print()

# cross validation done wrong
kf = KFold(n_splits=5, random_state=42)
accuracy = []
precision = []
recall = []
f1 = []
auc = []
X, y = SMOTE().fit_sample(X_train, y_train)
for train, test in kf.split(X, y):
    pipeline = make_pipeline(classifier(random_state=42))
    model = pipeline.fit(X[train], y[train])
    prediction = model.predict(X[test])

    accuracy.append(pipeline.score(X[test], y[test]))
    precision.append(precision_score(y[test], prediction))
    recall.append(recall_score(y[test], prediction))
    f1.append(f1_score(y[test], prediction))

print("done wrong mean of scores 5-fold:")
print("accuracy: {}".format(np.mean(accuracy)))
print("precision: {}".format(np.mean(precision)))
print("recall: {}".format(np.mean(recall)))
print("f1: {}".format(np.mean(f1)))


########################################################################################################################################################


 ####################################### PDF 2 EXCEL##############################
import io
import PyPDF2

from PyPDF2 import PdfFileReader

pdf = PdfFileReader(open('D:/Projects/dist_rainfall.PDF','rb'))
pages=pdf.getNumPages()
dict = pages.readlines()
for key in dict:
    print(key)
#PDFfilename = "D:/Projects/dist_rainfall.PDF" #filename of your PDF/directory where your PDF is stored
#pfr = open(pdf, "rb")
#dict = pfr.readlines()
#for key in pfr:
   # print(key)

########################################################################################################################################################################################

#################################### ROI Final.py##############################################################################

## THIS CONTAINS ilvl and NumId of EVERY TAG from DOCUMENT.xml

# define Roman Numbers (function)

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

import re
import os
import zipfile

 
import math



# file_path = "D:/ExxonMobil/all_documents/test/WCSD-000-IO-BSPDS-0002_r2"

def mainFunction(file_path):
    global pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    from collections import OrderedDict
    
    def isNumber(word):
        ''' Detect if strings like "55" etc. appear
        '''
        return word.isdigit() 
    
    
    def get_GP_in_Line(s1,prefix):
        GPList = s1.split(prefix)
        GPListString=""
        for index,val in enumerate(GPList):
            ##print i
            if index != 0 :
                GPListString = "".join([GPListString, " GP-",val.split(" ")[0]])
        return GPListString
    
    
    def write_roman(num,small=True):        
        roman = OrderedDict()
        roman[1000] = "M"
        roman[900] = "CM"
        roman[500] = "D"
        roman[400] = "CD"
        roman[100] = "C"
        roman[90] = "XC"
        roman[50] = "L"
        roman[40] = "XL"
        roman[10] = "X"
        roman[9] = "IX"
        roman[5] = "V"
        roman[4] = "IV"
        roman[1] = "I"
    
        def roman_num(num):
            for r in roman.keys():
                x, y = divmod(num, r)
                yield roman[r].lower() * x if small == True else roman[r].lower() * x 
                num -= (r * x)
                if num > 0:
                    roman_num(num)
                else:
                    break
        return "".join([a for a in roman_num(num)])
    
    # define alphabets function
    def getletters(num,lower=True):  # define letters if True = Lower if false Capital letters
        q,r = divmod(num,26) # beyond z
        ret = 'a' if lower == True else 'A'
        ret = chr(ord(ret) + r)
        if q != 0:
            ret = "".join([ret] * (q+1))
        return ret
    
    
    def extractInfoFromXML(filename,df):
        import xml.etree.ElementTree as ET
        import pandas as pd
        tree = ET.parse(filename)
        #cids2 = []
        count = 0
        count1 = 0
        TOCFOUND = False
        for c_node in tree.findall('.//p'):
            ##print(type(c_node.find("pPr/pStyle")))
            if type(c_node.find("pPr/pStyle")) != type(None):
                section = c_node.find("pPr/pStyle").attrib['val']
            else:
                section = "NOT FOUND"
            a=""
            # TODO: Temporary code of extracting GP before TOC, needs to be removed later
            if section == "TOC1":
                TOCFOUND=True
            count+=1
            for c_node2 in c_node.findall('.//r/t'):
                if a == "":
                    a = c_node2.text
    
                else:
                    a= "".join([a,c_node2.text])
        
            if a != "":
                GP=""
                if TOCFOUND == False:
                    #GP=""
                    data = get_GP_in_Line(a,"GP-")
                    GP = data
                    data = get_GP_in_Line(a,"GP ")
                    GP= " ".join([GP,data])
                    #print (GP)
                    
                if "Record of Change" in a and 'TOC' not in section:
                    if count1 > 0:
                        df['total'] = [count1]*len(df)
                    return df
                if True:      
                    count1+=1
                    #df = df.append(pd.DataFrame({'file_name':os.path.basename(filename).split('.')[0],'Row_ID':count,'num':count1,  
                    #                                 'section_name':section,'text':a.encode('utf-8')} , index=[count1] ))
                    df = df.append(pd.DataFrame({'file_name':os.path.basename(filename).split('.')[0],
                                                 'Row_Id':count,'num':count1,  
                                                 'section_name':section,
                                                 'text from ROI':a.encode('utf-8'),'GP':GP} ,index=[count1] ))
            df['total'] = [count1]*len(df)
        df.to_csv('ROI.csv', index = False)
        return df
                
           
    
    import os
    dir = file_path;#'C:/Users/20130407/Desktop/New folder/GP/gp030310_com'#/document.xml'
    os.chdir(dir)
    #def getDocumentXml(filename):
    import pandas as pd
    import xml.etree.ElementTree as ET
    tree = ET.parse('document.xml')
    tree1 = ET.parse('numbering.xml')
    tree2 = ET.parse('styles.xml')
    pstyle= []
    iLevel = []  
    numId = []
    fromStylesXMLTable = []
    df_final = pd.DataFrame()
    
    def mergeLeftInOrder(x, y, on=None):
        import numpy as np
        x = x.copy()
        x["Order"] = np.arange(len(x))
        z = x.merge(y, how='left', on=on).set_index("Order").ix[np.arange(len(x)), :]
        return z
    
    
    
    ### Stlyes, numID and ilvl from Styles.xml
    sStyles = []
    snumId = []
    silvl = []
    tree2 = ET.parse('styles.xml')
    for snode1 in tree2.findall('.//style'):
        for snode3 in snode1.findall('pPr'):
            check3 = snode3.find('numPr')
            if check3 is None:
                for snode4 in snode1.findall('basedOn'):
                    snumId.append((snode1.attrib['styleId'],snode4.attrib['val']))
                    silvl.append(str(0))
            else:
                for snode4 in snode3.findall('numPr'):
                    check= snode4.find('numId')
                    check1 = snode1.find('basedOn')
                    check3 = snode4.find('ilvl')
                    if check is None:
                        if check1 is None:
                            pass
                        else:
                            snumId.append((snode1.attrib['styleId'],check1.attrib['val']))
                            if check3 is None:
                                silvl.append('0')
                            else:
                                silvl.append(check3.attrib['val'])
                                
                    else: 
                        snumId.append((snode1.attrib['styleId'],check.attrib['val']))
                        if check3 is None:
                            silvl.append('0')
                        else:
                            silvl.append(check3.attrib['val'])
                        
                        
    numId_ilvl = []
    for i in range(0,len(snumId)): 
        numId_ilvl.append((snumId[i],silvl[i]))
        
    df_style = pd.DataFrame()
    count = 0
    for i in range(0,len(snumId)):
        count+=1
        df_style = df_style.append(pd.DataFrame({'style':numId_ilvl[i][0][0],'numId':numId_ilvl[i][0][1],'ilvl': numId_ilvl[i][1]}, index = [count]))
    #df_4 = df_4[['style', 'numId','ilvl']]
    a ={}
    if df_style.empty :
        sStyles_dict={}
    else:
        sStyles_dict = dict(zip(df_style['style'], df_style['numId']))
    
    
    
    
    
    def numIdRecursive(df,df_dict):
        for i in range(0,len(df)):
            if isNumber(str(df['numId'].iloc[i])) == True :
                pass
    
            else:
                for k in df_dict:
                                     
                    if df['numId'].iloc[i]  == k:
                        df['numId'].iloc[i] = df_dict[k]
                        
    if df_style.empty:
        pass
    else:
        for i in range(0,len(df_style)):
            numIdRecursive(df_style,sStyles_dict)
            
        import re
        for row in list(range(0,len(df_style))):
            df_style['numId'].iloc[row] = re.sub("[^0-9]", "", df_style['numId'].iloc[row])
        df_style['numId'].replace('', np.nan, inplace=True)
        df_style.dropna(subset=['numId'], inplace=True)
    
    
        
    sStyles = []
    nnumId_ilvl = []
    for i in range(0,len(df_style)):
        sStyles.append((df_style['style'].iloc[i]))
        nnumId_ilvl.append((df_style['numId'].iloc[i],df_style['ilvl'].iloc[i]))
        
    styles_dict = dict(zip(sStyles,nnumId_ilvl))
    
    
    
    fT = []
    count = 0
    tree = ET.parse('document.xml') 
    for node in tree.findall('.//p'):
        
        count+=1
        check1 = node.find('pPr/pStyle')
        if check1 is None: # if style is not there
            node_check = node.find('pPr/numPr/numId')
            if node_check is None:
                pass
            else:
                for node3 in node.findall('pPr/numPr/numId'):
                    for node4 in node.findall('pPr/numPr/ilvl'):
                        #count+=1
                        fT.append((count, 'No Style', (node3.attrib['val'], node4.attrib['val']) )) # change here count
        else:    
            for nodek2 in node.findall('pPr/pStyle'): # if style is present
                node_check = node.find('pPr/numPr/numId')
                if node_check is None:
                    if nodek2.attrib['val'] in styles_dict.keys():
                        fT.append((count,nodek2.attrib['val'], styles_dict[nodek2.attrib['val']] )) # change here count
                else:
                    for node3 in node.findall('pPr/numPr/numId'):
                        for node4 in node.findall('pPr/numPr/ilvl'):
                            fT.append((count, nodek2.attrib['val'], (node3.attrib['val'], node4.attrib['val']) )) # change here count
    
    
    
    
    
    def extract_from_xml(folder):
        import pandas as pd
        import os
        from os.path import isfile, join
        import xml.etree.ElementTree as ET
        df = pd.DataFrame(columns={'file_name','Row_Id' ,'section_name','text from ROI'})
        os.chdir(folder)
        xml_files = [f for f in os.listdir(folder) if isfile(join(folder, f)) and 'xml' in f]
        ##print xml_files
        for filename in xml_files:
            tree = ET.parse(filename)
            cids2 = []
            count = 0
            for c_node in tree.findall('.//p'):
                count+=1
                #for c_node1 in c_node.findall('pPr/pStyle'):
                if type(c_node.find("pPr/pStyle")) != type(None):
                    section = c_node.find("pPr/pStyle").attrib['val']
                else:
                     section = "NOT FOUND"
                a=""
                
                for c_node2 in c_node.findall('.//r/t'):
                    if a == "":
                        a = c_node2.text
        
                    else:
                        a= "".join([a,c_node2.text])
                        
                if ((a != "") and (not a.isspace())):
                    ##print (c_node1.attrib['val'],a.encode('utf-8'))
                    df1 = pd.DataFrame({'file_name':filename, 'Row_Id': count,'section_name':section,'text from ROI':a.encode('ascii','ignore')}, index=[0] )
                    df=df.append(df1)
        
        #df.to_csv('ROI_from_xml.csv', mode='a', header=False)
        return df
    
    
    
    fT_1 = []  
    count = 0
    for node in tree.findall('.//p/pPr'):
        count+=1
        check1 = node.find('pStyle') 
        if check1 is None:
            
            fT_1.append((count, 'NA'))
            
        else:
            for nodek1 in node.findall('pStyle'):
                #count+=1
                fT_1.append((count,nodek1.attrib['val']))
                
    fT1_df = pd.DataFrame()
    count = 0
    for i in range(0,len(fT_1)):
        count+=1
        fT1_df = fT1_df.append(pd.DataFrame({'Row_Id':fT_1[i][0],'Style':fT_1[i][1]},index=[count]))
                
    
    #fT_df = mergeLeftInOrder(fT1_df,fT2_df, on = ['Row_Id','Style'])    # this will give the list of styles with and without the numid, ilvl and absnumID's
    
    
    
    ###### this is all correct and working do not delete
                       
    fT_df = pd.DataFrame()
    count=0
    for i in range(0,len(fT)):
        count+=1
        fT_df = fT_df.append(pd.DataFrame({'Row_Id':fT[i][0],'Style':fT[i][1],'numId':fT[i][2][0],'ilvl':fT[i][2][1]},index=[count]))
                
                
                
                
    ###from NUMBERING.XML
    import pandas as pd
    import xml.etree.ElementTree as ET
    tree1 = ET.parse('numbering.xml')
    tree2 = ET.parse('styles.xml')
    numTable = [] 
    for node in tree1.findall('.//num'):
        for node1 in node.findall('abstractNumId'):
            check = node.find('lvlOverride/startOverride')
            if check is None:
                numTable.append((node.attrib['numId'],node1.attrib['val'],''))
            else:
                            
                numTable.append((node.attrib['numId'],node1.attrib['val'],check.attrib['val']))
    
    numId_df = pd.DataFrame(numTable, columns = ['numId','abstractNumId','startOverride']) 
    
    
    abstractTab = []
    for node in tree1.findall('.//abstractNum'):
        for node1 in node.findall('lvl'):
            for node2 in node1.findall('numFmt'):
                for node3 in node1.findall('lvlText'):
                    abstractTab.append((node.attrib['abstractNumId'],node1.attrib['ilvl'],node2.attrib['val'],node3.attrib['val']))
    abstractNum_df = pd.DataFrame(abstractTab, columns = ['abstractNumId','ilvl','numFmt','lvlText'])
    
    #Defineing DataFrames to store thr final Result
    
    
    df11 = ''
    final_df = ''
    df1 = ''
    if 'numId' in fT_df:   
        df11 = mergeLeftInOrder(fT_df, numId_df, on = 'numId')
        final_df = mergeLeftInOrder(df11,abstractNum_df, on = ['abstractNumId','ilvl'])
        # working for GP
        df1 = final_df
        df1.dropna(subset=['abstractNumId'], inplace=True)
    
    
    
    
    
    
    def incrementNumber(score,diffIndex):      
       #score = "5.1"
       #diffIndex = 1
       #score = "2.5"
       #diffIndex = 2
       
       if(diffIndex>0):
           while diffIndex>0:
               score = score[0:score.rfind(".")]
               diffIndex = diffIndex-1; 
       if(str(score).count(".")>0):
         
         score = incrementFloat(score,0)
          # score =   str(dummyvalue)#+tempstrScore       
       else:
        
         ##print "$$$$$$$$$$$$$$$"
         score = int(score)+1   
       return score
    
    def incrementFloat(score,diffIndex):
        if( diffIndex<1 and (str(score).count(".")>0)):
            tempstrScore = str(score)     
            dummyvalue = int(tempstrScore[score.rfind(".")+1:len(tempstrScore)])        
            dummyvalue = dummyvalue+1        
            score = score[0:score.rfind(".")]
            score = score+"."+str(dummyvalue)       
        else:
            while(diffIndex>0):
                score = str(score)+".1"
                diffIndex = diffIndex-1;
        return score
    
    def appendFloat(score):
        score = str(score)+".1"
        return score
    
    def removeZero(score):
        #score = dfObj['score']
        str(score).find(".")
        if(str(score).count(".0")>0):
            scoreStr = str(score)
            score= scoreStr[0:len(scoreStr)-2] 
        return score
    
    def uniqueVal(seq): # Order preserving
      seen = set()
      return [x for x in seq if x not in seen and not seen.add(x)]
    
    
    def appendSplCharacter(currentdfObj):
        ##print currentdfObj 
        #if not ((str(currentdfObj['numFmt']) is str('decimal')) or (str(currentdfObj['numFmt']) is str('bullet'))):    
        if not ((str(currentdfObj['numFmt']) == str('bullet')) or (str(currentdfObj['numFmt']) == str('decimal'))):
            lvlText = str(currentdfObj['lvlText']).encode('utf-8')
            currentdfObj['realNumber'] = currentdfObj['realNumber']+lvlText[-1]
        elif(str(currentdfObj['numFmt']) == str('decimal')):
            if(str(currentdfObj['lvlText']).find(".0")>=0):
                currentdfObj['realNumber'] = currentdfObj['realNumber']+".0"
        return currentdfObj
    
    
    def getFormatedNumber(x,numFmt):
        if (numFmt.lower() == "lowerletter"):
            return getletters(int(x) -1,True)
        elif (numFmt.lower() == "upperletter"):
            return getletters(int(x) -1,False)
        elif (numFmt.lower() == "lowerroman"):
            return write_roman(int(x) ,True)
        elif (numFmt.lower() == "upperroman"):
            return write_roman(int(x) ,False) 
        elif(numFmt.lower() == "bullet"):
            return '' 
        else:
            return x
        
    def getLastIndexValForLowerCase(dataFrameObj):
    
        numFmt = str(dataFrameObj['numFmt'])
        score = str(dataFrameObj['score'])
        if(numFmt.lower() == str('lowerletter') or numFmt.lower() == str('lowerroman') or 
           numFmt.lower() == str('upperletter') or numFmt.lower() == str('upperroman')):   
            if(score.count(".")>=0): 
                score = score[score.rfind(".")+1:len(score)]
        return score
    
    def numberingGenarator(df1):
        #**************Initializing Varialble ****************
        numIdArray = []
        abstractNum_ilvl_combo = []
        abstractNum_ilvl_combo_dict = {}
        dataframe = df1
        SIZE = len(dataframe)  
        
        dataframe['score'] = '0'
        # dataframe['numbers'] = '0'
        dataframe['realNumber'] = '0'
        new_dict = {}
        uniqueEmpInt = uniqueVal(dataframe['abstractNumId'])
    
    
        #Assigning new unique values to 
        for unique in uniqueEmpInt:
            new_dict[unique] = dataframe.iloc[0]
            
    
         
        #************************The Complete Logic*******************    
        for j in range(0,SIZE):
            currentdfObj = dataframe.iloc[j]
            score=1
            if(j==0):
                currentdfObj['score'] = score
            else:
                #if currentdfObj['Row_Id'] = '159':
                    
                if  (currentdfObj['startOverride']=='1' and (int(currentdfObj['numId']) not in numIdArray)):
                    key = str(currentdfObj['abstractNumId'])+"_"+str(currentdfObj['ilvl'])
                   # if(key in abstractNum_ilvl_combo):
                    #    currentdfObj['score'] = abstractNum_ilvl_combo_dict[key]['score']
                   # else:
                    currentdfObj['score'] = 1    
                             
                else:
                    prevSimilarObj = new_dict[currentdfObj['abstractNumId']];
                                   
                    if(int(currentdfObj['ilvl'])>0):                
                        if(int(prevSimilarObj['ilvl'])==int(currentdfObj['ilvl'])):
                            score = prevSimilarObj['score']
                           # score = incrementFloat(score)
                            if(str(score).count(".")>0):              
                                diffIndex = int(currentdfObj['ilvl']) - int(prevSimilarObj['ilvl'])                                 
                                score = prevSimilarObj['score']
                                score = incrementFloat(score,diffIndex)
                            else:
                                diffIndex = int(prevSimilarObj['ilvl']) - int(currentdfObj['ilvl'])
                                score = incrementNumber(score,diffIndex)
                            currentdfObj['score'] = score                  
                        elif(int(prevSimilarObj['ilvl'])<int(currentdfObj['ilvl'])):    
                            diffIndex = int(currentdfObj['ilvl']) - int(prevSimilarObj['ilvl'])                                 
                            score = prevSimilarObj['score']
                            score = incrementFloat(score,diffIndex)
                            currentdfObj['score'] = score
                        elif(int(prevSimilarObj['ilvl'])>int(currentdfObj['ilvl'])):                                      
                            score = prevSimilarObj['score']
                            # score = math.floor(float(score))+1  
                            diffIndex = int(prevSimilarObj['ilvl']) - int(currentdfObj['ilvl'])
                            score = incrementNumber(score,diffIndex)
                            currentdfObj['score'] = score
                    else:                          
                        score = prevSimilarObj['score']
                        # score = math.floor(float(score))+1 
                        diffIndex = int(prevSimilarObj['ilvl']) - int(currentdfObj['ilvl'])
                        score = incrementNumber(score,diffIndex)
                        currentdfObj['score'] = score              
                #elif(prevdfObj['abstractNumId']!=currentdfObj['abstractNumId']):
            currentdfObj['score']=removeZero(currentdfObj['score'])
            #currentdfObj['final_numbering_test']=getFormatedNumber((str(currentdfObj['score']).encode('utf-8'),currentdfObj['numFmt'].encode('utf-8')))
            final_numbering_test_score = getLastIndexValForLowerCase((currentdfObj))
            final_numbering_test_nymfmt = str(currentdfObj['numFmt'])
            # currentdfObj['numbers']=getFormatedNumber(final_numbering_test_score,final_numbering_test_nymfmt)
            currentdfObj['realNumber']=getFormatedNumber(final_numbering_test_score,final_numbering_test_nymfmt)
            currentdfObj = appendSplCharacter(currentdfObj)
            
            dataframe.iloc[j] = currentdfObj
            new_dict[currentdfObj['abstractNumId']] = currentdfObj
            key = str(currentdfObj['abstractNumId'])+"_"+str(currentdfObj['ilvl'])
            abstractNum_ilvl_combo_dict[key] = currentdfObj
            numIdArray.append(int(currentdfObj['numId']))
            abstractNum_ilvl_combo.append(str(currentdfObj['abstractNumId'])+"_"+str(currentdfObj['ilvl']))
            
        return dataframe
    
    
    
    
    final = ''
    if(len(df1)>0):
        final = numberingGenarator(df1)
              
    
    roi_df = extract_from_xml(file_path)
    
    # merge the paragraph with numbering using row_id
    RROI_d_df = ''
    RROI_df = pd.DataFrame()
    if(len(final)>0):
        RROI_d_df = mergeLeftInOrder(roi_df,final, on = ['Row_Id'])
        
        RROI_df = RROI_d_df[['Row_Id','section_name','file_name','text from ROI','numFmt','lvlText','realNumber']].copy()
    
    # remove if 'nan is ther in paragraph and convert into .csv
    
    RROI_final_df = pd.DataFrame()
    for i in range(0,len(RROI_df)):
        if str(RROI_df['realNumber'].iloc[i]) == 'nan':
            RROI_final_df = RROI_final_df.append(pd.DataFrame({'Row_Id':RROI_df['Row_Id'].iloc[i], 'section_name':RROI_df['section_name'].iloc[i],'file_name': os.path.dirname(RROI_df['file_name'].iloc[i]).split('/')[-1], 'text from ROI': RROI_df['text from ROI'].iloc[i], 'RROI': str(RROI_df['text from ROI'].iloc[i])}, index = [0]))
        else:
            RROI_final_df = RROI_final_df.append(pd.DataFrame({'Row_Id':RROI_df['Row_Id'].iloc[i], 'section_name':RROI_df['section_name'].iloc[i],'file_name': RROI_df['file_name'].iloc[i], 'text from ROI': RROI_df['text from ROI'].iloc[i], 'RROI': str(RROI_df['realNumber'].iloc[i])+' '+str(RROI_df['text from ROI'].iloc[i])}, index = [0]))
    
    RROI_final_df['document_name'] = file_path.replace('\\','/').split('/')[-1]
    
    RROI_final_df.to_csv('RROI_final_df.csv', encoding = 'utf-8',index = False) 


    
def wColonFromTag(docFilePath, docXZip, xmlPath):
    fname1 = os.path.splitext(os.path.basename(docFilePath))[0]
    if not os.path.exists(fname1):
        os.mkdir(fname1)
    dXmlPath = os.path.join(os.path.dirname(docFilePath),fname1,os.path.basename(xmlPath))
    #print (dXmlPath)
    j = docXZip.extract(xmlPath)
    #print (j)
    inFile1 = open(j, 'r')
    data1 = inFile1.readlines()
    s = ""
    A = []
    B = []
    for i in data1:
        replaced = re.sub('w:', s, i)
        A.append(replaced)

    inFile1.close()
    #xmlPath
    
    inFile1 = open(dXmlPath, 'w')
    inFile1.writelines(A)
    inFile1.close()
    return os.path.join(os.getcwd(),dXmlPath)

def generateSupportingFile(directoryPath,fileName):
    myPwd = os.getcwd()
    os.chdir(directoryPath)
    fileToHandle = fileName
    try:
        document = zipfile.ZipFile(fileToHandle)
        docDotXML = 'word/document.xml'
        styleDotXML = 'word/styles.xml'
        numberingDotXML = 'word/numbering.xml'
    
        if (docDotXML and  styleDotXML and numberingDotXML) in  document.namelist() :
            docDotXML = wColonFromTag(fileToHandle, document, docDotXML)
            styleDotXML = wColonFromTag(fileToHandle, document, styleDotXML)
            numberingDotXML = wColonFromTag(fileToHandle, document, numberingDotXML)
            print (docDotXML, styleDotXML, numberingDotXML)
            os.chdir(myPwd)
    except (zipfile.BadZipfile):
        print fileToHandle
    
def creatingAllSupportingFiles(directoryPath):
    from os import walk
    
    #os.walk(directory)
    os.listdir(directoryPath)
    files = []
    for (dirpath, dirnames, filenames) in walk(directoryPath):        
        files.extend(filenames)
        break
    
    for fileName in files:        
        generateSupportingFile(directoryPath,fileName)
        
def processAlltheFiles(directoryPath):
    #os.walk(directory)
    os.listdir(directoryPath)
    count =0;
    df = pd.DataFrame()
    for x in os.walk(directoryPath):
        if(count>0):
            print "@@@@@@@@@@@@@@"
            if "word"  not in x[0]: 
                print x[0]
                df = df.append(mainFunction(x[0]))
                
        count = count+1
    return df
        
directoryPath = 'D:/ExxonMobil/Final_All_Docs/GP/GP2/gp_2.2'
creatingAllSupportingFiles(directoryPath)
df_final = processAlltheFiles(directoryPath)

df_final.to_csv(directoryPath+'/RROI_final_df.csv', encoding = 'utf-8',index = False) 


########################################################################################################################################################
##################################### TABLE Format##########################################################################

# coding: utf-8


"""
Created on Nov 09 2017 19:50:03

@author: 20122134
"""

from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer
from nltk.tag import StanfordNERTagger
import re
import nltk
from collections import OrderedDict
from collections import Iterable
import numpy as np
import pandas as pd
import scipy as sp
import re
SOME_FIXED_SEED = 42
np.random.seed(SOME_FIXED_SEED)





class regexLetters(object):
    def __init__(self):
        pass

    def process(self,letterList):
        #print (letterList)
        taggedLetters = {'ALPHA':[],'DIGIT':[],'PUNCSPEC':[]}
        for i in letterList:
            if i.isalpha():
                #print ("ALPHABET : [{}]".format(i))
                taggedLetters['ALPHA'].append(i)
            elif i.isdigit():
                #print ("DIGIT : [{}]".format(i))
                taggedLetters['DIGIT'].append(i)
            else:
                #print ("PUNCT-SPECIAL: [{}]".format(i))
                taggedLetters['PUNCSPEC'].append(i)
        return taggedLetters        
        

class regexWords(object):
    def __init__(self):
        self.letterProcessor = regexLetters()
    
    def tagWord(self,word,taggedLetters):
        #TODO : Evaluate all options of tagging from general English and then on top of that apply domain rules
        taggedWords = OrderedDict()
        
        #print("WORD, TaggedLetters")
        #print(word,taggedLetters)
        if len(taggedLetters['DIGIT']) == 0:
            if len(taggedLetters['PUNCSPEC']) == 0 and len(taggedLetters['ALPHA']) !=0:
                #PURE words
                #What they could be : Author, company name, others ?
                #Check in dictionary
                #Check NER? better do NER in RegexSentence before sending it here
                #Check Table
                #Check names etc
                # TODO: Enable this code after installing pyenchant
                taggedWords[word] = "English"
                '''
                if engDict.check(word):
                    #print ("In English DICTIONARY")
                    #if i not in taggedWords.keys():
                    taggedWords[word] = "English-DICT"
                    #taggedWords[i].append()
                else:
                    #print ("NOT in English DICTIONARY")
                    taggedWords[word] = "English-Non-DICT"
                    # TODO: Many possibilities, find out and tag
                #print ("PURE word :{}".format(word))
                '''
            if len(taggedLetters['PUNCSPEC']) != 0 and len(taggedLetters['ALPHA']) !=0:
                #Words mixed with special characters
                #Where is special characters? 
                #    In the end?  Are they comma, full stops, can they be removed
                #    what they are? 
                #    Are they meaningless
                #    Can they be removed
                    taggedWords[word] = "English-PUNCSPEC"
                    #print ("MIXED word :{}".format(word))
                    #if word[-1] == ':':
                        #taggedWords[word] = "Key"
                        
            if len(taggedLetters['PUNCSPEC']) != 0 and len(taggedLetters['ALPHA']) ==0:
                # PURE PUNCSPEC
                    taggedWords[word] = "PUNCSPEC"
                    #print ("MIXED word :{}".format(word))
                    
                    #if word[-1] == ':':
                        #taggedWords[word] = "Key"
            
        elif len(taggedLetters['DIGIT']) != 0:
            if len(taggedLetters['PUNCSPEC']) == 0 and len(taggedLetters['ALPHA']) ==0:
                # DIGITS
                # What they could be, pure numbers? or more
                #print ("DIGITS :{}".format(word))
                taggedWords[word] = "DIGIT"
                # TODO: Tag it further, date(May be outside in RegexSentence class), quantities (it could be well depth or more)
            elif len(taggedLetters['PUNCSPEC']) != 0 and len(taggedLetters['ALPHA']) ==0:
                # DIGITS
                # What they could be, Amount, Serial Numbers, Dates or more
                #print ("DIGITS-SPEC :{}".format(word))
                taggedWords[word] = "DIGIT-PUNCSPEC"
            elif len(taggedLetters['PUNCSPEC']) == 0 and len(taggedLetters['ALPHA']) !=0:
                # DIGITS
                # What they could be, Amount, Serial Numbers, Dates or more
                #print ("DIGITS-ALPHA :{}".format(word))
                taggedWords[word] = "English-DIGIT"
            elif len(taggedLetters['PUNCSPEC']) != 0 and len(taggedLetters['ALPHA']) !=0:
                # DIGITS
                # What they could be -  Serial Numbers, Dates or more
                #print ("DIGITS-SPEC-ALPHA :{}".format(word))
                taggedWords[word] = "English-DIGIT-PUNCSPEC"
        return taggedWords

    def process(self,wordList):
        #print(wordList)
        #for word in wordList:
            #self.letterTokens = list(word)
            #print(self.letterTokens)
            #print (self.letterProcessor.process(self.letterTokens))
            
        spacesIndex = np.where(np.array(wordList) == ' ')[0]
        #print(spacesIndex)
        #wordListSepSpace = wordList.split(' ')
        #print (wordListSepSpace)
        prevI = 0
        taggedWords = OrderedDict()
        #taggedWords = []
        for i in spacesIndex:
            
            #print (wordList[prevI:i])
            taggedLetters = self.letterProcessor.process(wordList[prevI:i])
            word = "".join(wordList[prevI:i])
            #print (len(taggedWords))
            #print("Words to tag is :{}".format(word))
            #taggedWords.append(tagWord(word,taggedLetters))
            
            taggedWord = self.tagWord(word,taggedLetters)
            if len(taggedWords) == 0:
                taggedWords = taggedWord
            else:
                taggedWords.update(taggedWord)
            #print(taggedWords)
            prevI=i+1
        if len(spacesIndex) == 0:
            taggedLetters = self.letterProcessor.process(wordList)
            word = "".join(wordList)
            #print("Single Word to tag is :{}".format(word))
            taggedWord = self.tagWord(word,taggedLetters)
            taggedWords = taggedWord
        else:
            taggedLetters = self.letterProcessor.process(wordList[prevI:len(wordList)+1])
            word = "".join(wordList[prevI:len(wordList)+1])
            #print (len(taggedWords))
            #print("LAST: Single Word to tag is :{}".format(word))
            #taggedWords.append(tagWord(word,taggedLetters))
            
            taggedWord = self.tagWord(word,taggedLetters)
            #print(len(taggedWords))
            if len(taggedWords) == 0:
                taggedWords = taggedWord
            else:
                taggedWords.update(taggedWord)
            
        #print ("WordTagger: TaggedWords are {}".format(taggedWords))
        return (taggedWords)

# RegexSentence
class regexSentence(object):
    def __init__(self,kvSplitterList):
        #self.a = a
        #self.tok = MosesTokenizer()
        self.kvSplitterList = kvSplitterList
        #self.tok = sent_tokenize()
        #self.deTok = MosesDetokenizer()
        self.wordProcessor = regexWords()
        pass
    
    def process(self,sentence):
        #self.tokens = self.tok.tokenize(sentence)
        #self.tokens = sent_tokenize(sentence)
        #self.tokens = word_tokenize(sentence)
        self.tokens = list(sentence)
        #print (sentence)
        taggedWords = self.wordProcessor.process(self.tokens)
        
        taggedSentence = OrderedDict({'OTHER':[],'KEY':[],'VALUE':[], 'KEY_WITHOUT_VALUE':[]})
        # Key value tagging
        #print("Tagged words are: {}".format(taggedWords))
        for index,value in enumerate(taggedWords):
            #print ("INDEX :{}, VALUE :{}".format(index,value))
            #print ("taggedWords[index] {}".format(taggedWords[index]))
            #value = list(taggedWords.keys()[index])
            if index == 0:
                valueToAdd = list(taggedWords.keys())[index]
                if valueToAdd[-1] in self.kvSplitterList:
                    # if there is no value
                    if len(taggedWords) == 1:
                        taggedSentence['KEY_WITHOUT_VALUE'] = [valueToAdd]
                        return (taggedSentence)
                    taggedSentence['KEY'] = [valueToAdd]
                    # TODO : To handle if multiple key-value pair come in single line
                    
                    #print("Length")
                    #print(len(taggedWords))
                    valueToAdd = list(taggedWords.keys())[index+1:len(taggedWords)]
                    taggedSentence['VALUE'] = valueToAdd
                    #print("A")
                    #print(taggedSentence)
                    return (taggedSentence)
                else:
                    taggedSentence['OTHER'].append(valueToAdd)
                    #print("B")
                    #print(taggedSentence)
            else:
                if value in self.kvSplitterList or value[0] in self.kvSplitterList:
                    #print(taggedWords)
                    # if there is no value
                    if index+1 == len(taggedWords) and value in self.kvSplitterList:
                        valueToAdd = list(taggedWords.keys())[0:index+1]
                        #print(valueToAdd)
                        taggedSentence['OTHER'] = []
                        taggedSentence['KEY_WITHOUT_VALUE'] = valueToAdd
                        return (taggedSentence)
                    #print(taggedWords[0:index][0])
                    valueToAdd = list(taggedWords.keys())[0:index]
                    taggedSentence['KEY'] = valueToAdd
                    #taggedSentence['KEY'].append(valueToAdd)
                    taggedSentence['OTHER'] = []
                    # TODO : To handle if multiple key-value pair come in single line
                    
                    valueToAdd = list(taggedWords.keys())[index:len(taggedWords)]
                    taggedSentence['VALUE'] = valueToAdd
                    #print("C")
                    #print(taggedSentence)
                    return (taggedSentence)
                else:
                    valueToAdd = list(taggedWords.keys())[index]
                    taggedSentence['OTHER'].append(valueToAdd)
                    #print("D")
                    #print(taggedSentence)
                    
            #print (taggedSentence)
        return (taggedSentence)
    
class regexSection(object):
    def __init__(self,kvSplitterList):
        self.kvSplitterList = kvSplitterList
        self.sentenceProcessor = regexSentence(kvSplitterList=kvSplitterList)
        pass
    
    def process(self,section):
        taggedSection = None
        for sentence in section:
            if isinstance(section,list) == True:
                sentence = "".join(sentence)
            taggedSentence = self.sentenceProcessor.process(sentence)
            if type(taggedSection) == type(None):
                taggedSection = [taggedSentence]
            else:
                taggedSection.append(taggedSentence)
                #print(type(taggedSentence),len(taggedSentence),len(taggedSection),type(taggedSection))
            #print(3)
            #print (taggedSection)
        return taggedSection


#############   Input Data Frame
#df = pd.read_excel("D:/Projects/Exxon/Exxon_NLP/Output_TableFormat.xlsx")

#########
def col_rows(df):
    df = df.fillna(' ')
    obj={}    
    ls=df['table_number'].unique()
    for i in ls:
        table=df[df['table_number']==i]
        keys_1=[]
        #for key,value in table['ocr_output']:
            #val=[]
            #for j in table['table_row_number']:
        vals=table[table['table_number']==i]
        valnum=vals['table_row_number'].unique()
        value= vals[vals['table_row_number']==valnum[0]]['ocr_output'].tolist()
        keys=table[table['table_number']==i]
        keyse=keys[keys['table_row_number']==valnum[1]]['ocr_output'].tolist()
        for h,f in zip(value,keyse):
            obj.setdefault(h,[]).append(f) 
    return obj
col_rows(df)




def returnNoneOrString(x):
    regI = regexSentence(kvSplitterList=[''])
    se = col_rows(df)
    dicts = [str(key)+' : '+ str(val) for key in se for val in se[key]] 
    rS = regexSection(kvSplitterList=[':'])
    taggedSection = rS.process(dicts)
    #print (taggedSection)
    if len(x) == 0:
        return None
    return " ".join(x)

a = [[]]
for idx,val in enumerate(dicts):
    #print(taggedSection[idx])
    a.append([str(val), 
              returnNoneOrString(taggedSection[idx]['KEY']),
              returnNoneOrString(taggedSection[idx]['VALUE']), 
              returnNoneOrString(taggedSection[idx]['KEY_WITHOUT_VALUE']),
              returnNoneOrString(taggedSection[idx]['OTHER'])
             ])
			 
#### Output Data Frame  			 
			 
Output = pd.DataFrame(a,columns=['text','key','value','key_without_value','other'])



Output

##########################################################################################################################################################

################################### TABLE TEXT Format #################################################


# coding: utf-8


"""
Created on Nov 09 2017 19:50:03

@author: 
"""

from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer
from nltk.tag import StanfordNERTagger
import re
import nltk
from collections import OrderedDict
from collections import Iterable
import numpy as np
import pandas as pd
import scipy as sp
import re
SOME_FIXED_SEED = 42
np.random.seed(SOME_FIXED_SEED)


# In[6]:


class regexLetters(object):
    def __init__(self):
        pass

    def process(self,letterList):
        #print (letterList)
        taggedLetters = {'ALPHA':[],'DIGIT':[],'PUNCSPEC':[]}
        for i in letterList:
            if i.isalpha():
                #print ("ALPHABET : [{}]".format(i))
                taggedLetters['ALPHA'].append(i)
            elif i.isdigit():
                #print ("DIGIT : [{}]".format(i))
                taggedLetters['DIGIT'].append(i)
            else:
                #print ("PUNCT-SPECIAL: [{}]".format(i))
                taggedLetters['PUNCSPEC'].append(i)
        return taggedLetters        
        

class regexWords(object):
    def __init__(self):
        self.letterProcessor = regexLetters()
    
    def tagWord(self,word,taggedLetters):
        #TODO : Evaluate all options of tagging from general English and then on top of that apply domain rules
        taggedWords = OrderedDict()
        
        #print("WORD, TaggedLetters")
        #print(word,taggedLetters)
        if len(taggedLetters['DIGIT']) == 0:
            if len(taggedLetters['PUNCSPEC']) == 0 and len(taggedLetters['ALPHA']) !=0:
                #PURE words
                #What they could be : Author, company name, others ?
                #Check in dictionary
                #Check NER? better do NER in RegexSentence before sending it here
                #Check Table
                #Check names etc
                # TODO: Enable this code after installing pyenchant
                taggedWords[word] = "English"
                '''
                if engDict.check(word):
                    #print ("In English DICTIONARY")
                    #if i not in taggedWords.keys():
                    taggedWords[word] = "English-DICT"
                    #taggedWords[i].append()
                else:
                    #print ("NOT in English DICTIONARY")
                    taggedWords[word] = "English-Non-DICT"
                    # TODO: Many possibilities, find out and tag
                #print ("PURE word :{}".format(word))
                '''
            if len(taggedLetters['PUNCSPEC']) != 0 and len(taggedLetters['ALPHA']) !=0:
                #Words mixed with special characters
                #Where is special characters? 
                #    In the end?  Are they comma, full stops, can they be removed
                #    what they are? 
                #    Are they meaningless
                #    Can they be removed
                    taggedWords[word] = "English-PUNCSPEC"
                    #print ("MIXED word :{}".format(word))
                    #if word[-1] == ':':
                        #taggedWords[word] = "Key"
                        
            if len(taggedLetters['PUNCSPEC']) != 0 and len(taggedLetters['ALPHA']) ==0:
                # PURE PUNCSPEC
                    taggedWords[word] = "PUNCSPEC"
                    #print ("MIXED word :{}".format(word))
                    
                    #if word[-1] == ':':
                        #taggedWords[word] = "Key"
            
        elif len(taggedLetters['DIGIT']) != 0:
            if len(taggedLetters['PUNCSPEC']) == 0 and len(taggedLetters['ALPHA']) ==0:
                # DIGITS
                # What they could be, pure numbers? or more
                #print ("DIGITS :{}".format(word))
                taggedWords[word] = "DIGIT"
                # TODO: Tag it further, date(May be outside in RegexSentence class), quantities (it could be well depth or more)
            elif len(taggedLetters['PUNCSPEC']) != 0 and len(taggedLetters['ALPHA']) ==0:
                # DIGITS
                # What they could be, Amount, Serial Numbers, Dates or more
                #print ("DIGITS-SPEC :{}".format(word))
                taggedWords[word] = "DIGIT-PUNCSPEC"
            elif len(taggedLetters['PUNCSPEC']) == 0 and len(taggedLetters['ALPHA']) !=0:
                # DIGITS
                # What they could be, Amount, Serial Numbers, Dates or more
                #print ("DIGITS-ALPHA :{}".format(word))
                taggedWords[word] = "English-DIGIT"
            elif len(taggedLetters['PUNCSPEC']) != 0 and len(taggedLetters['ALPHA']) !=0:
                # DIGITS
                # What they could be -  Serial Numbers, Dates or more
                
                taggedWords[word] = "English-DIGIT-PUNCSPEC"
        return taggedWords

    def process(self,wordList):
        #print(wordList)                   
        spacesIndex = np.where(np.array(wordList) == ' ')[0]
        
        prevI = 0
        taggedWords = OrderedDict()
        #taggedWords = []
        for i in spacesIndex:
            
            #print (wordList[prevI:i])
            taggedLetters = self.letterProcessor.process(wordList[prevI:i])
            word = "".join(wordList[prevI:i])
            
            taggedWord = self.tagWord(word,taggedLetters)
            if len(taggedWords) == 0:
                taggedWords = taggedWord
            else:
                taggedWords.update(taggedWord)
            #print(taggedWords)
            prevI=i+1
        if len(spacesIndex) == 0:
            taggedLetters = self.letterProcessor.process(wordList)
            word = "".join(wordList)
            #print("Single Word to tag is :{}".format(word))
            taggedWord = self.tagWord(word,taggedLetters)
            taggedWords = taggedWord
        else:
            taggedLetters = self.letterProcessor.process(wordList[prevI:len(wordList)+1])
            word = "".join(wordList[prevI:len(wordList)+1])
            
            
            taggedWord = self.tagWord(word,taggedLetters)
           
            if len(taggedWords) == 0:
                taggedWords = taggedWord
            else:
                taggedWords.update(taggedWord)
            
        #print ("WordTagger: TaggedWords are {}".format(taggedWords))
        return (taggedWords)

# RegexSentence
class regexSentence(object):
    def __init__(self,kvSplitterList):
       
        self.kvSplitterList = kvSplitterList
       
        self.wordProcessor = regexWords()
        pass
    
    def process(self,sentence):
      
        self.tokens = list(sentence)
        #print (sentence)
        taggedWords = self.wordProcessor.process(self.tokens)
        
        taggedSentence = OrderedDict({'OTHER':[],'KEY':[],'VALUE':[], 'KEY_WITHOUT_VALUE':[]})
        # Key value tagging
        
        for index,value in enumerate(taggedWords):
            
            if index == 0:
                valueToAdd = list(taggedWords.keys())[index]
                if valueToAdd[-1] in self.kvSplitterList:
                    # if there is no value
                    if len(taggedWords) == 1:
                        taggedSentence['KEY_WITHOUT_VALUE'] = [valueToAdd]
                        return (taggedSentence)
                    taggedSentence['KEY'] = [valueToAdd]
                    # TODO : To handle if multiple key-value pair come in single line
                    
                    
                    valueToAdd = list(taggedWords.keys())[index+1:len(taggedWords)]
                    taggedSentence['VALUE'] = valueToAdd
                    
                    return (taggedSentence)
                else:
                    taggedSentence['OTHER'].append(valueToAdd)
                    
            else:
                if value in self.kvSplitterList or value[0] in self.kvSplitterList:
                   
                    if index+1 == len(taggedWords) and value in self.kvSplitterList:
                        valueToAdd = list(taggedWords.keys())[0:index+1]
                       
                        taggedSentence['OTHER'] = []
                        taggedSentence['KEY_WITHOUT_VALUE'] = valueToAdd
                        return (taggedSentence)
                    
                    valueToAdd = list(taggedWords.keys())[0:index]
                    taggedSentence['KEY'] = valueToAdd
                    
                    taggedSentence['OTHER'] = []
                   
                    valueToAdd = list(taggedWords.keys())[index:len(taggedWords)]
                    taggedSentence['VALUE'] = valueToAdd
                    
                    return (taggedSentence)
                else:
                    valueToAdd = list(taggedWords.keys())[index]
                    taggedSentence['OTHER'].append(valueToAdd)
                  
        return (taggedSentence)
    
class regexSection(object):
    def __init__(self,kvSplitterList):
        self.kvSplitterList = kvSplitterList
        self.sentenceProcessor = regexSentence(kvSplitterList=kvSplitterList)
        pass
    
    def process(self,section):
        taggedSection = None
        for sentence in section:
            if isinstance(section,list) == True:
                sentence = "".join(sentence)
            taggedSentence = self.sentenceProcessor.process(sentence)
            if type(taggedSection) == type(None):
                taggedSection = [taggedSentence]
            else:
                taggedSection.append(taggedSentence)
                
        return taggedSection





sam_table = pd.read_excel("D:/Projects/Exxon/Exxon_NLP/Output_TableFormat.xlsx")
def col_rows(sam_table):
    sam_table = sam_table.fillna(' ')
    obj={}    
    ls=sam_table['table_number'].unique()
    for i in ls:
        table=sam_table[sam_table['table_number']==i]
        keys_1=[]
        vals=table[table['table_number']==i]
        valnum=vals['table_row_number'].unique()
        value= vals[vals['table_row_number']==valnum[0]]['ocr_output'].tolist()
        keys=table[table['table_number']==i]
        keyse=keys[keys['table_row_number']==valnum[1]]['ocr_output'].tolist()
        for h,f in zip(value,keyse):
            obj.setdefault(h,[]).append(f) 
    return obj
col_rows(sam_table)




def returnNoneOrString(x):
    regI = regexSentence(kvSplitterList=[''])
    se = col_rows(sam_table)
    dicts = [str(key)+' : '+ str(val) for key in se for val in se[key]] 
    rS = regexSection(kvSplitterList=[':'])
    taggedSection = rS.process(dicts)
    
    if len(x) == 0:
        return None
    return " ".join(x)

a = [[]]
for idx,val in enumerate(dicts):
    
    a.append([str(val), 
              returnNoneOrString(taggedSection[idx]['KEY']),
              returnNoneOrString(taggedSection[idx]['VALUE']), 
              returnNoneOrString(taggedSection[idx]['KEY_WITHOUT_VALUE']),
              returnNoneOrString(taggedSection[idx]['OTHER'])
             ])
myDf = pd.DataFrame(a,columns=['text','key','value','key_without_value','other'])


############### Sai Krishna Code Starts Here


def get_substring_indices(text1, q):
    result = [i for i in range(len(text1)) if text1.startswith(q, i)]
    k = []
    #print(result)
    for i in result:
        if i==0 or i==len(text1)-len(q):
            k.append(i)
        else:
            if text1[i-1]== ' ' and text1[i+len(q)]==' ':
                k.append(i)
    return k
    s ={'well':['Well no','Well'] ,'Total Depth':['Depth'],'DocAuthor':['Name'],'Top Log Interval':['Top','top loginterval','Interval']
       ,'scale':['Scale'],'latitude' :['Lat','Latitude'],'lon':['Lon','Longitude']}
    #key =[]
    #value =[]
    for p,q in enumerate(df['ocr_output']):
        q = str(q)
        #q  = q.lower()
        q = q.replace(':'," ")
        col_keys =""
        col_values=""
        for i in s:
            key = []
            value=[]
            for bal in s[i]:
                #print(bal)
                ind = get_substring_indices(q, bal)
                new_list = [x+len(bal) for x in ind]
                if len(new_list) != 0:
                    for j,e in enumerate(new_list):
                        key.append(bal)
                        if(j == len(new_list)-1):
                            value.append(q[e:])
                        else:
                            value.append(q[e:new_list[j+1]-len(bal)])
            #print(key,value)
            if len(key)!=0 and len(col_keys)!=0:
                col_keys = col_keys +"|"+ "|".join(key)
                col_values = col_values +"|"+ "|".join(value)
            elif len(col_keys)==0:
                col_keys = col_keys + "|".join(key)
                col_values = col_values + "|".join(value)

        df.loc[p,'key'] = aa
        df.loc[p,'value'] = bb

####################################################################################################################################################

from __future__ import print_function
import cv2
import pandas as pd
import numpy as np
image = cv2.imread('D:/Projects/Exxon/Exxon_NLP/-3.png')

gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gs_image = cv2.fastNlMeansDenoising(gs_img, None, 65, 5, 21)

rows, cols = gs_image.shape

# Assuming that print stray marks occur in the page boundaries.
gs_image_padded = gs_image[0:rows - 0, 0:]

# Inverting the colors
gs_image_padded = 255 - gs_image_padded
gs_image_padded[gs_image_padded < 128] = 0 
gs_image_padded[gs_image_padded > 128] = 255
#print (gs_image_padded.shape)
def lineExtract(image):
        def getHorizontalLimits(img):
            INFINITY = 100000
            mn, mx = INFINITY, 0
            for r in img:
                non_zero = np.where(r == 255)
                mn = min(mn, np.min(non_zero))
                mx =  max(mx, np.max(non_zero))
            return mn, mx

        image[image < 128] = 0
        image[image > 128] = 255
        rows, cols = image.shape
        lines = [] #Stores the lines
        line_of_text = []
        start_row=[]
        end_row=[]
        line_no=[]
        start_col=[]
        end_col=[]
        row_index = 0
        line_no = 1
        tracking = False
        this_line_start_index = 0

        for row in image:
#Checking if atleast one nonzero pixel exist in row
            if not tracking:
                this_line_start_index = row_index

            if np.count_nonzero(row) > 0:
                line_of_text.append(row)
                tracking = True
            elif np.array(line_of_text).shape[0] > 25:
                start_col, end_col = getHorizontalLimits(np.array(line_of_text))
                lines.append({"start_row":this_line_start_index,
                    "end_row":row_index,
                    "line_no":line_no,
                    "start_col":start_col,
                    "end_col":end_col,
                    #"pixels":np.array(line_of_text)
                    })
                line_of_text = []
                line_no+=1
                tracking = False
                row_index += 1
        return lines
lines = lineExtract(gs_image_padded)
sama=[]
df = pd.DataFrame()
for line in lines:
#for key,values in sorted(lines.items()):
    #list_keys = [ k for k in line.values()]
    #sama.append(line.copy())
    #print(list(line.values()))
    print(list(line.keys()))
    df = pd.DataFrame(line.values(), columns=['start_row','end_row','line_no','start_col','end_col'])
    #print(df)
        #print(type(s.))
	#first_row.append(this_line_start_index) 
    #start=line['start_row']
    #print("starting row: ", line["start_row"])
    #print("starting row: ", start)
    #print("ending row: ", line["end_row"])
    #print("starting col: ", line["start_col"])
    #print("ending col: ", line["end_col"])
    #print("Line Number: ", line["line_no"])
    #print("pixels: ", line["pixels"])









