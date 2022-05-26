import numpy as np

def is_different_script(word1, word2):
    return word1.isascii()^word2.isascii()

def get_i_index(sentence) -> float:
    sp=0
    tokens=sentence.split(' ')
    if len(tokens)>=2:
        for j in range(1, len(tokens)):
            this_word = tokens[j]
            prev_word = tokens[j-1]
            if is_different_script(prev_word, this_word):
                sp+=1
        return sp/(len(tokens)-1)
    else:
        return 0

SPI_CUTOFF=0.33

def get_spi_bucket_tag(s):
    spi_score = get_i_index(s)
    if spi_score==0:
        spi_bucket_tag = "spi_zero"
    elif spi_score<=SPI_CUTOFF:
        spi_bucket_tag = "spi_lo"
    else:
        spi_bucket_tag = "spi_hi"
    return spi_bucket_tag

def get_spi_bucket_accuracy(target, prediction):
    spi_bucket_prediction = get_spi_bucket_tag(prediction)
    spi_bucket_target = get_spi_bucket_tag(target)
    return float(spi_bucket_prediction==spi_bucket_target)

def spi_bucket_accuracy(targets, predictions):
    total_accuracy = sum(
            get_spi_bucket_accuracy(target, prediction)
            for target, prediction in zip(targets, predictions))
    return {"spi_bucket_accuracy": total_accuracy/len(predictions)}

def spi_correlation(targets, predictions):
    spi_target = [get_i_index(sentence) for sentence in targets]
    spi_prediction = [get_i_index(sentence) for sentence in predictions]
    correlation = np.corrcoef(spi_target, spi_prediction)[0,1]
    return {"spi_correlation":correlation}

