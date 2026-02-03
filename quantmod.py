from quantmod.derivatives import OptionData



if __name__ == "__main__":
    print("Quantmod derivatives module loaded successfully.")
    # Example usage of OptionData
    option_data = OptionData(symbol="NIFTY", expiration="2025-Aug-28")
    print(option_data)

    # # Example usage of OptionChain
    # option_chain = OptionChain(symbol="AAPL")
    # print(option_chain.get_chain())

    # # Example usage of OptionPrice
    # option_price = OptionPrice(option_data)
    # print(f"Option Price: {option_price.calculate()}")  