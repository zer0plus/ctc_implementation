#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define MAX_WORD_LENGTH 100


// dict like structure to hold a word and its frequency
typedef struct {
    char word[MAX_WORD_LENGTH];
    int count;
} WordFrequency;


// used for qsort to compare words and their counts
int compare(const void *a, const void *b) {
    return ((WordFrequency *)b)->count - ((WordFrequency *)a)->count;
}


// removes trailing punctuations to count words properly.
void remove_punctuation(char *word) {
    int len = strlen(word);
    if (ispunct(word[len - 1])) {
        word[len - 1] = '\0';
    }
}


char **find_frequent_words(const char *path, int32_t n) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return NULL;
    }

    int max_words = 1000;  // Initial capacity
    WordFrequency *word_counts = (WordFrequency *)malloc(max_words * sizeof(WordFrequency));
    if (word_counts == NULL) {
        printf("Failed to allocate memory.\n");
        fclose(file);
        return NULL;
    }

    int num_words = 0;

    char line[1000];
    while (fgets(line, sizeof(line), file)) {
        // splits space, tab, newline, and carriage return characters
        char *token = strtok(line, " \t\n\r");

        while (token != NULL) {
            remove_punctuation(token);
            
            // convert the word to lowercase
            for (int i = 0; token[i]; i++) {
                token[i] = tolower(token[i]);
            }

            // check if the word already exists in the array
            int found = 0;
            for (int i = 0; i < num_words; i++) {
                if (strcmp(word_counts[i].word, token) == 0) {
                    word_counts[i].count++;
                    found = 1;
                    break;
                }
            }

            // if word not found, add it to the array
            if (!found) {
                if (num_words >= max_words) {
                    max_words *= 2;  // Double the capacity of max words
                    WordFrequency *temp = (WordFrequency *)realloc(word_counts, max_words * sizeof(WordFrequency));
                    if (temp == NULL) {
                        printf("Failed to reallocate memory.\n");
                        free(word_counts);
                        fclose(file);
                        return NULL;
                    }
                    word_counts = temp;
                }
                strcpy(word_counts[num_words].word, token);
                word_counts[num_words].count = 1;
                num_words++;
            }

            token = strtok(NULL, " \t\n\r");
        }
    }

    fclose(file);

    // sort the word counts array in descending order
    qsort(word_counts, num_words, sizeof(WordFrequency), compare);

    // alloc memory for the array of strings
    char **frequent_words = (char **)malloc(n * sizeof(char *));
    if (frequent_words == NULL) {
        printf("Failed to allocate memory.\n");
        free(word_counts);
        return NULL;
    }

    // copy n most frequent words to the array of strings
    for (int i = 0; i < n && i < num_words; i++) {
        frequent_words[i] = strdup(word_counts[i].word);
    }

    free(word_counts);

    return frequent_words;
}


int main() {
    const char *filename = "tf_Shakespeare.txt";
    int n = 5;
    char **frequent_words = find_frequent_words(filename, n);
    if (frequent_words != NULL) {
        printf("%d most frequent words in the TensorFlow Shakespeare dataset:\n", n);
        for (int i = 0; i < n; i++) {
            printf("%d) %s\n", i+1, frequent_words[i]);
            free(frequent_words[i]);
        }
        free(frequent_words);
    }
    return 0;
}