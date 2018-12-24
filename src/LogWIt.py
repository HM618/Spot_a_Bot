import pyspark as ps
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, Tokenizer, OneHotEncoder, StringIndexer, VectorAssembler
import bleach
from pyspark.sql.functions import *

spark = (
        ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("lecture")
        .getOrCreate()
        )

sc = spark.sparkContext

it_all = spark.read.csv('only_italians.csv',
                         header=True)

regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words")

add_stopwords = ["RT","rt","/","@","the",'ios8','http://t.co','a', 'abbastanza', 'abbia', 'abbiamo', 'abbiano', 'abbiate', 'accidenti', 'ad', 'adesso', 'affinche', 'agl', 'agli', 'ahime', 'ahimã¨', 'ahimè', 'ai', 'al', 'alcuna', 'alcuni', 'alcuno', 'all', 'alla', 'alle', 'allo', 'allora', 'altre', 'altri', 'altrimenti', 'altro', 'altrove', 'altrui', 'anche', 'ancora', 'anni', 'anno', 'ansa', 'anticipo', 'assai', 'attesa', 'attraverso', 'avanti', 'avemmo', 'avendo', 'avente', 'aver', 'avere', 'averlo', 'avesse', 'avessero', 'avessi', 'avessimo', 'aveste', 'avesti', 'avete', 'aveva', 'avevamo', 'avevano', 'avevate', 'avevi', 'avevo', 'avrai', 'avranno', 'avrebbe', 'avrebbero', 'avrei', 'avremmo', 'avremo', 'avreste', 'avresti', 'avrete', 'avrà', 'avrò', 'avuta', 'avute', 'avuti', 'avuto', 'basta', 'ben', 'bene', 'benissimo', 'berlusconi', 'brava', 'bravo', 'buono', 'c', 'casa', 'caso', 'cento', 'certa', 'certe', 'certi', 'certo', 'che', 'chi', 'chicchessia', 'chiunque', 'ci', 'ciascuna', 'ciascuno', 'cima', 'cinque', 'cio', 'cioe', 'cioã¨', 'cioè', 'circa', 'citta', 'città', 'cittã', 'ciã²', 'ciò', 'co', 'codesta', 'codesti', 'codesto', 'cogli', 'coi', 'col', 'colei', 'coll', 'coloro', 'colui', 'come', 'cominci', 'comprare', 'comunque', 'con', 'concernente', 'conciliarsi', 'conclusione', 'consecutivi', 'consecutivo', 'consiglio', 'contro', 'cortesia', 'cos', 'cosa', 'cosi', 'cosã¬', 'così', 'cui', 'd', 'da', 'dagl', 'dagli', 'dai', 'dal', 'dall', 'dalla', 'dalle', 'dallo', 'dappertutto', 'davanti', 'degl', 'degli', 'dei', 'del', 'dell', 'della', 'delle', 'dello', 'dentro', 'detto', 'deve', 'devo', 'di', 'dice', 'dietro', 'dire', 'dirimpetto', 'diventa', 'diventare', 'diventato', 'dopo', 'doppio', 'dov', 'dove', 'dovra', 'dovrà', 'dovrã', 'dovunque', 'due', 'dunque', 'durante', 'e', 'ebbe', 'ebbero', 'ebbi', 'ecc', 'ecco', 'ed', 'effettivamente', 'egli', 'ella', 'entrambi', 'eppure', 'era', 'erano', 'eravamo', 'eravate', 'eri', 'ero', 'esempio', 'esse', 'essendo', 'esser', 'essere', 'essi', 'ex', 'fa', 'faccia', 'facciamo', 'facciano', 'facciate', 'faccio', 'facemmo', 'facendo', 'facesse', 'facessero', 'facessi', 'facessimo', 'faceste', 'facesti', 'faceva', 'facevamo', 'facevano', 'facevate', 'facevi', 'facevo', 'fai', 'fanno', 'farai', 'faranno', 'fare', 'farebbe', 'farebbero', 'farei', 'faremmo', 'faremo', 'fareste', 'faresti', 'farete', 'farà', 'farò', 'fatto', 'favore', 'fece', 'fecero', 'feci', 'fin', 'finalmente', 'finche', 'fine', 'fino', 'forse', 'forza', 'fosse', 'fossero', 'fossi', 'fossimo', 'foste', 'fosti', 'fra', 'frattempo', 'fu', 'fui', 'fummo', 'fuori', 'furono', 'futuro', 'generale', 'gente', 'gia', 'giacche', 'giorni', 'giorno', 'giu', 'già', 'giã', 'gli', 'gliela', 'gliele', 'glieli', 'glielo', 'gliene', 'governo', 'grande', 'grazie', 'gruppo', 'ha', 'haha', 'hai', 'hanno', 'ho', 'i', 'ie', 'ieri', 'il', 'improvviso', 'in', 'inc', 'indietro', 'infatti', 'inoltre', 'insieme', 'intanto', 'intorno', 'invece', 'io', 'l', 'la', 'lasciato', 'lato', 'lavoro', 'le', 'lei', 'li', 'lo', 'lontano', 'loro', 'lui', 'lungo', 'luogo', 'là', 'lã', 'ma', 'macche', 'magari', 'maggior', 'mai', 'male', 'malgrado', 'malissimo', 'mancanza', 'marche', 'me', 'medesimo', 'mediante', 'meglio', 'meno', 'mentre', 'mesi', 'mezzo', 'mi', 'mia', 'mie', 'miei', 'mila', 'miliardi', 'milioni', 'minimi', 'ministro', 'mio', 'modo', 'molta', 'molti', 'moltissimo', 'molto', 'momento', 'mondo', 'mosto', 'nazionale', 'ne', 'negl', 'negli', 'nei', 'nel', 'nell', 'nella', 'nelle', 'nello', 'nemmeno', 'neppure', 'nessun', 'nessuna', 'nessuno', 'niente', 'no', 'noi', 'nome', 'non', 'nondimeno', 'nonostante', 'nonsia', 'nostra', 'nostre', 'nostri', 'nostro', 'novanta', 'nove', 'nulla', 'nuovi', 'nuovo', 'o', 'od', 'oggi', 'ogni', 'ognuna', 'ognuno', 'oltre', 'oppure', 'ora', 'ore', 'osi', 'ossia', 'ottanta', 'otto', 'paese', 'parecchi', 'parecchie', 'parecchio', 'parte', 'partendo', 'peccato', 'peggio', 'per', 'perche', 'perchã¨', 'perchè', 'perché', 'percio', 'perciã²', 'perciò', 'perfino', 'pero', 'persino', 'persone', 'perã²', 'però', 'piedi', 'pieno', 'piglia', 'piu', 'piuttosto', 'piã¹', 'più', 'po', 'pochissimo', 'poco', 'poi', 'poiche', 'possa', 'possedere', 'posteriore', 'posto', 'potrebbe', 'preferibilmente', 'presa', 'press', 'prima', 'primo', 'principalmente', 'probabilmente', 'promesso', 'proprio', 'puo', 'pure', 'purtroppo', 'puã²', 'può', 'qua', 'qualche', 'qualcosa', 'qualcuna', 'qualcuno', 'quale', 'quali', 'qualunque', 'quando', 'quanta', 'quante', 'quanti', 'quanto', 'quantunque', 'quarto', 'quasi', 'quattro', 'quel', 'quella', 'quelle', 'quelli', 'quello', 'quest', 'questa', 'queste', 'questi', 'questo', 'qui', 'quindi', 'quinto', 'realmente', 'recente', 'recentemente', 'registrazione', 'relativo', 'riecco', 'rispetto', 'salvo', 'sara', 'sarai', 'saranno', 'sarebbe', 'sarebbero', 'sarei', 'saremmo', 'saremo', 'sareste', 'saresti', 'sarete', 'sarà', 'sarã', 'sarò', 'scola', 'scopo', 'scorso', 'se', 'secondo', 'seguente', 'seguito', 'sei', 'sembra', 'sembrare', 'sembrato', 'sembrava', 'sembri', 'sempre', 'senza', 'sette', 'si', 'sia', 'siamo', 'siano', 'siate', 'siete', 'sig', 'solito', 'solo', 'soltanto', 'sono', 'sopra', 'soprattutto', 'sotto', 'spesso', 'srl', 'sta', 'stai', 'stando', 'stanno', 'starai', 'staranno', 'starebbe', 'starebbero', 'starei', 'staremmo', 'staremo', 'stareste', 'staresti', 'starete', 'starà', 'starò', 'stata', 'state', 'stati', 'stato', 'stava', 'stavamo', 'stavano', 'stavate', 'stavi', 'stavo', 'stemmo', 'stessa', 'stesse', 'stessero', 'stessi', 'stessimo', 'stesso', 'steste', 'stesti', 'stette', 'stettero', 'stetti', 'stia', 'stiamo', 'stiano', 'stiate', 'sto', 'su', 'sua', 'subito', 'successivamente', 'successivo', 'sue', 'sugl', 'sugli', 'sui', 'sul', 'sull', 'sulla', 'sulle', 'sullo', 'suo', 'suoi', 'tale', 'tali', 'talvolta', 'tanto', 'te', 'tempo', 'terzo', 'th', 'ti', 'titolo', 'torino', 'tra', 'tranne', 'tre', 'trenta', 'triplo', 'troppo', 'trovato', 'tu', 'tua', 'tue', 'tuo', 'tuoi', 'tutta', 'tuttavia', 'tutte', 'tutti', 'tutto', 'uguali', 'ulteriore', 'ultimo', 'un', 'una', 'uno', 'uomo', 'va', 'vai', 'vale', 'vari', 'varia', 'varie', 'vario', 'verso', 'vi', 'via', 'vicino', 'visto', 'vita', 'voi', 'volta', 'volte', 'vostra', 'vostre', 'vostri', 'vostro', 'ã¨', 'è']

stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])

notbot = it_all.filter('label=0')
bot = it_all.filter('label=1')
it_all.count()
#1921228
notbot.count()
#311384
bot.count()
#1609944
sampleRatio = notbot.count()/ it_all.count()
#0.1620

botsample = bot.sample(True, sampleRatio)
# botsample.count() 260472

balanced = notbot.unionAll(botsample)

#balanced.filter('label=0').count() #55%

#balanced.filter('label=1').count() #45%

balanced = balanced.withColumn("label", it_all["label"].cast(IntegerType()))

pipelineFit = pipeline.fit(balanced)
trans_data = pipelineFit.transform(balanced)
# trans_data.show(5)

lr = LogisticRegression(maxIter=10,
                        regParam=0.001,
                        featuresCol='features',
                        labelCol='label',
                        predictionCol='prediction',
                        probabilityCol='probability')

train_df, test_df = trans_data.randomSplit([0.7, 0.3], seed = 100)
# print("Training Dataset Count: " + str(train_df.count()))
# print("Test Dataset Count: " + str(test_df.count()))

logistic_model = lr.fit(train_df)

predictions = logistic_model.transform(test_df)

#predictions[['text', 'label', 'filtered', 'features', 'probability', 'prediction']].show(5)

test_df = test_df.select('text', 'words', 'filtered', 'features')
test_out = logistic_model.transform(test_df)

import matplotlib.pyplot as plt
import numpy as np

beta = np.sort(logistic_model.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
# plt.show()

trainingSummary = logistic_model.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig('ROC Curve')
#plt.show()
# print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)
# 0.9817511240257981
