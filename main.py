from rag_bot.retrieval import ask_question


def main():

    print("GitHub Repository Chatbot")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask: ")

        if question.lower() == "exit":
            break

        print()
        print("ai-bot : " ,ask_question(question))
        print()


if __name__ == "__main__":
    main()