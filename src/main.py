from genre_utils import load_netflix_data, clean_netflix_data, count_genre_pairs


def main():
    df = load_netflix_data()
    df = clean_netflix_data(df)

    while True:
        print("\n===== NETFLIX GENRE ANALYZER =====")
        print("1. View top 10 genre combinations")
        print("2. Exit")

        choice = input("Enter your choice (1-2): ")

        if choice == "1":
            pairs = count_genre_pairs(df["genres"])

            print("\nTop 10 Genre Combinations:")
            for pair, count in pairs.most_common(10):
                print(f"{pair[0]} & {pair[1]}: {count} titles")

        elif choice == "2":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
