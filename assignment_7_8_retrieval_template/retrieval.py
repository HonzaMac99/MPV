import numpy as np, time, datetime, scipy.io, os, pdb, pickle
from scipy.sparse import csr_matrix
import PIL.Image
from utils import get_pts_in_box, draw_bbox, vis_results, get_A_matrix_from_geom, get_query_data, get_shortlist_data
np.set_printoptions(linewidth=400, threshold=np.inf)


def create_db(image_visual_words, num_visual_words, idf):
    """
    create the image database with an inverted file represented as a sparse matrix. 
    the sparse matrix has dimension number_of_visual_words x number_of_images
    the stored representation should be l2 normalized

    image_visual_words: list of arrays indicating the visual words present in each image
    num_visual_words: total number of visual words in the visual codebook
    idf: array with idf weights per visual word
    return -> 
    db: sparse matrix representing the inverted file 
    """

    # changing sparsity structure is expensive --> convert at the end
    # db = csr_matrix((num_visual_words, len(image_visual_words)))

    db = np.zeros((num_visual_words, len(image_visual_words)))

    for i, img_vis_w in enumerate(image_visual_words):
        img_vec = np.bincount(img_vis_w, minlength=num_visual_words) * idf
        if len(img_vis_w) > 0:  # normalize only nonzero vectors
            img_vec /= np.linalg.norm(img_vec)
        db[:, i] = img_vec

    db = csr_matrix(db)

    return db


def get_idf(image_visual_words, num_visual_words):
    """
    Calculate the IDF weight for visual word

    image_visual_words: list of arrays indicating the visual words present in each image
    num_visual_words: total number of visual words in the visual codebook
    return -> 
    idf: array with idf weights per visual word
    """

    N = len(image_visual_words)       # total number of images
    n_i = np.zeros(num_visual_words)  # number of occurrences of each visual word in all images

    for i in range(N):
        n_i[np.unique(image_visual_words[i])] += 1

    idf = np.zeros_like(n_i)
    n_i_positive = n_i.nonzero()
    idf[n_i_positive] = np.log(N/n_i[n_i_positive])  # for n_i = 0, we set idf to be 0

    return idf


def retrieve(db, query_visual_words, idf):
    """
    Search the database with a query, sort images base on similarity to the query. 
    Returns ranked list of image ids and the corresponding similarities

    db: image database
    query_visual_words: array with visual words of the query image
    idf: idf weights
    return -> 
    ranking: sorted list of image ids based on similarities to the query
    sim: sorted list of similarities
    """

    query_vec = np.bincount(query_visual_words.flatten(), minlength=idf.shape[0]) * idf
    query_vec /= np.linalg.norm(query_vec)

    scores = query_vec.T @ db

    ranking = np.argsort(-scores)
    sim = -np.sort(-scores)

    return ranking, sim


def get_tentative_correspondences(query_visual_words, shortlist_visual_words):
    """
    query_visual_words: 1D array with visual words of the query 
    shortlist_visual_words: list of 1D arrays with visual words of top-ranked images 
    return -> 
    correspondences: list of lists of correspondences
    """

    correspondences = []

    # loop over the provided list of DB images
    for i, img_visual_words in enumerate(shortlist_visual_words):

        corr = []
        for j, query_vis_w in enumerate(query_visual_words):
            img_idx = np.c_[np.where(img_visual_words == query_vis_w)]
            query_idx = np.ones_like(img_idx) * j
            corr_arr = np.hstack((query_idx, img_idx))
            if np.sum(corr_arr.shape) > 0:
                corr += corr_arr.tolist()

        # append correspondences for image i
        correspondences.append(corr)

    return correspondences


def ransac_affine(query_geometry, shortlist_geometry, correspondences, inlier_threshold):
    """

    query_geometry: 2D array with geometry of the query
    shortlist_geometry: list of 2D arrays with geometry of top-ranked images
    correspondences: list of lists of correspondences
    inlier_threshold: threshold for inliers of the spatial verification
    return -> 
    inlier_counts: 1D array with number of inliers per image
    transformations: 3D array with the transformation per image
    """

    K = len(shortlist_geometry)
    transformations = np.zeros((K, 3, 3))
    inliers_counts = np.zeros((K, ))

    for k in range(K):
        best_score = 0
        A_best = None

        corr = np.array(correspondences[k])
        N = len(corr)

        for n in range(N):
            q_id = corr[n, 0]
            d_id = corr[n, 1]

            Aq = get_A_matrix_from_geom(query_geometry[q_id]) # shape of local feature from the query
            Ad = get_A_matrix_from_geom(shortlist_geometry[k][d_id]) # shape of local feature from DB image

            # estimate transformation hypothesis A and the number of inliers - your code
            A = Ad @ np.linalg.inv(Aq)

            x1 = shortlist_geometry[k][corr[:, 1], :2]
            x2 = query_geometry[corr[:, 0], :2]
            x2_ones = np.ones((x2.shape[0], 1))

            x2_proj = A @ np.hstack((x2, x2_ones)).T
            x2_proj = x2_proj[:2].T
            x_diff = np.linalg.norm(x1 - x2_proj, axis=1)

            number_of_inliers = np.sum(x_diff > inlier_threshold)

            if number_of_inliers > best_score:
                best_score = number_of_inliers
                A_best = A

        inliers_counts[k] = best_score
        transformations[k] = A_best

    return transformations, inliers_counts


def search_spatial_verification(query_visual_words, query_geometry, candidatelist_visual_words, candidatelist_geometries, inlier_threshold):
    """

    query_visual_words: 1D array with visual words of the query 
    query_geometry: 2D array with geometry of the query
    candidatelist_visual_words: list of 1D arrays with visual words of top-ranked images 
    candidatelist_geometry: list of 2D arrays with geometry of top-ranked images
    inlier_threshold: threshold for inliers of the spatial verification
    inlier_counts: 1D array with number of inliers per image
    transformations: 3D array with the transformation per image
    """
    corrs = get_tentative_correspondences(query_visual_words, candidatelist_visual_words)    
    transformations, inliers_counts = ransac_affine(query_geometry, candidatelist_geometries, corrs, inlier_threshold)
    return inliers_counts, transformations



### ========================================================
def main():

    # set to True for the second part - spatial verif.
    include_lab_assignment_2 = True

    with open('data/mpv_lab_retrieval_data.pkl', 'rb') as handle:
        p = pickle.load(handle)     

    visual_words = p['visual_words']
    geometries = p['geometries']
    img_names = p['img_names']
    img_names = ['imgs/'+x+'.jpg' for x in img_names]
    print(len(img_names))
    num_visual_words = 50000+1  # for the codebook we used to generate the provided visual words

    # spatial verification parameters
    shortlist_size = 50
    inlier_threshold = 8

    t = time.time()
    idf = get_idf(visual_words, num_visual_words)
    db = create_db(visual_words, num_visual_words, idf)
    print("DB created in {:.5}s".format(time.time()-t))

    q_id = 367 # pick a query     
    t = time.time()
    ranked_img_ids, scores = retrieve(db, visual_words[q_id], idf)
    print("query performed in {:.3f}s".format(time.time() - t))
    print(ranked_img_ids[:10], scores[:10])

    if include_lab_assignment_2:
        bbox_xyxy = [350, 200, 700, 600] # pick a bounding box
        query_visual_words_inbox, query_geometry_inbox = get_query_data(visual_words, geometries, q_id, bbox_xyxy)
        t = time.time()
        ranked_img_ids, scores = retrieve(db, query_visual_words_inbox, idf)
        print("query-cropped performed in {:.3f}s".format(time.time() - t))
        print(ranked_img_ids[:10], scores[:10])

        shortlist_ids = ranked_img_ids[:shortlist_size]  # apply SP only to most similar images
        shortlist_visual_word, shortlist_geometry = get_shortlist_data(visual_words, geometries, shortlist_ids)

        t = time.time()
        scores_sp, transformations = search_spatial_verification(query_visual_words_inbox, query_geometry_inbox, shortlist_visual_word, shortlist_geometry, inlier_threshold)
        print("spatial verification performed in {:.3f}s".format(time.time() - t))

        idxs = np.argsort(-scores_sp)
        scores_sp = scores_sp[idxs]
        transformations = transformations[idxs]
        top_img_ids = ranked_img_ids[idxs]
        print(top_img_ids[:10], scores_sp[:10])

        # will create fig.png - check it out
        vis_results(img_names, q_id, bbox_xyxy, top_img_ids, scores_sp, transformations)


if __name__ == '__main__':
    main()

