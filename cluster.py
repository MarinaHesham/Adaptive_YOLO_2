import os
import argparse
import numpy as np 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default='data/coco/val2014/', help="Path of labels directory")
    parser.add_argument("--num_classes", type=int, default=80, help="Number of classes")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of object clusters")
    parser.add_argument("--output", type=str, default="cluster.data", help="Path to output the clusters")
    opt = parser.parse_args()

    directory = opt.labels
    num_classes = opt.num_classes
    num_clusters = opt.num_clusters

    ##### Compute the frequency based adjacency Matrix #####

    objects_adjacency_matrix = np.zeros((num_classes,num_classes))

    # Loop on Images Labels files
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # Loop on Objects inside each Image
            objects = []
            with open(directory + filename, "r") as a_file:
                for line in a_file:
                    stripped_line = line.split(" ")
                    objects.append(int(stripped_line[0]))
            for obj1 in objects:
                for obj2 in objects:
                    objects_adjacency_matrix[obj1,obj2] += 1
                    objects_adjacency_matrix[obj2,obj1] += 1

    np.savetxt("co-occurence_adjacency_matrix.csv", objects_adjacency_matrix, delimiter=",", fmt='%d')
    
    #### Cluster the objects according to the adjacency matrix #####

    classes_names = []
    with open("data/coco.names", "r") as file_names:
        for line in file_names:
            classes_names.append(line[:-1])
    classes_names = np.asarray(classes_names)

    # objects_adjacency = np.genfromtxt('proximity_adjacency_matrix_train.csv', delimiter=',',dtype=float)

    class_repetition = []
    for i in range(objects_adjacency_matrix.shape[0]):
        class_repetition.append(objects_adjacency_matrix[i,i])
    max_5_classes = np.argsort(class_repetition)[-1*num_clusters-1:-1]

    print(max_5_classes)
    print(classes_names[max_5_classes])

    clusters = {}
    for i in range(objects_adjacency_matrix.shape[0]):
        if i in max_5_classes:
                continue
        nearest = max_5_classes[0]
        for j in max_5_classes:
            if objects_adjacency_matrix[i,j] > objects_adjacency_matrix[i,nearest]:
                nearest = j
        if nearest in clusters.keys():
            clusters[nearest].append(i)
        else:
            clusters[nearest] = [i]
            
    print(clusters)

    to_be_deleted = []
    for cluster in clusters.keys():
        if len(clusters[cluster]) < 5:
            print(cluster)
            unmatched = [cluster]
            for i in clusters[cluster]:
                unmatched.append(i)
            to_be_deleted.append(cluster)
            for i in unmatched:
                nearest = np.argsort(class_repetition)[0]
                for cluster in clusters.keys():
                    if cluster in to_be_deleted:
                        continue
                    if objects_adjacency_matrix[i,cluster] > objects_adjacency_matrix[i,nearest]:
                        nearest = cluster
                clusters[nearest].append(i)

    for item in to_be_deleted:
        clusters.pop(item,None)

    print(clusters)

    with open("clusters.data", 'w') as f:
        for key in clusters.keys():
            f.write("%s"%(key))
            for cls_ in clusters[key]:
                f.write(",%s"%(cls_))
            f.write("\n")

    for key in clusters.keys():
        print(">>>>")
        print(classes_names[key])
        for item in clusters[key]:
            print(classes_names[item])
    
    # objects_adjacency[objects_adjacency>5000] = 5000
    figure = plt.figure() 
    axes = figure.add_subplot(111) 

    caxes = axes.matshow(objects_adjacency_matrix, interpolation ='nearest') 
    figure.colorbar(caxes)

    axes.set_xticks(np.arange(0,80))
    axes.set_yticks(np.arange(0,80))

    axes.set_xticklabels(classes_names, rotation=90) 
    axes.set_yticklabels(classes_names, rotation=0) 
    
    plt.show()
