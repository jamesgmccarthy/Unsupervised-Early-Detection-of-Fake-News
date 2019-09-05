"""
File to just pull everything together
"""
import createDatasets
import preprocessing
import CreateEmbeddings
import TopicClustering
import IDEC


def main(create_data=True, preprocess=True, create_embeddings=True, topic_clustering=True, fnd=True, under_sampled_fnd=False, topic_fnd=True):
    if create_data is True:
        createDatasets.main()
    if preprocess is True:
        sizes = ['large','small']
        preprocessing.main(sizes, 'clustering')
        preprocessing.main(sizes, 'fnd')
    if create_embeddings is True:
        print("Creating Embedding for clustering")
        CreateEmbeddings.main(retrain_model=True, dataset='clustering')
        print("Creating Embeddings for Fake News Detection")
        CreateEmbeddings.main(retrain_model=True, dataset='fnd')
    if topic_clustering is True:
        TopicClustering.main(force=True, dataset='politifact')
        IDEC.main(exp='topic', dataset='politifact')
        TopicClustering.main(force=True, dataset='gossipcop')
        IDEC.main(exp='topic', dataset='gossipcop')
    if fnd is True:
        IDEC.main(exp='fnd', dataset='politifact')
        IDEC.main(exp='fnd', dataset='gossipcop')

    if under_sampled_fnd is True:
        IDEC.main(exp='fnd', dataset='politifact',
                  topics=False, under_sample=True)
        IDEC.main(exp='fnd', dataset='gossipcop',
                  topics=False, under_sample=True)
    if topic_fnd is True:
        IDEC.main(exp='fnd', dataset='politifact',
                  topics=True, under_sample=False)
        IDEC.main(exp='fnd', dataset='gossipcop',
                  topics=True, under_sample=False)
    if under_sampled_fnd is True and topic_fnd is True:
        IDEC.main(exp='fnd', dataset='politifact',
                  topics=True, under_sample=True)
        IDEC.main(exp='fnd', dataset='gossipcop',
                  topics=True, under_sample=True)


if __name__ == '__main__':
    main(create_data=True, preprocess=True,
         create_embeddings=True, topic_clustering=True, fnd=True, topic_fnd=True, under_sampled_fnd=True)
