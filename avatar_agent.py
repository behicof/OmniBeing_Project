class TradingAvatarAgent:
    """
    The TradingAvatarAgent class simulates a trading agent with basic functionalities for interaction and trading execution.
    """

    def __init__(self, name="OmniBot"):
        """
        Initializes the TradingAvatarAgent with a name, position, and action log.

        Parameters:
        name (str): The name of the trading agent. Default is "OmniBot".
        """
        self.name = name
        self.position = 0
        self.action_log = []

    def speak(self, message):
        """
        Simulates the agent speaking with a given message.

        Parameters:
        message (str): The message to be spoken by the agent.

        Returns:
        str: The formatted message spoken by the agent.
        """
        return f"{self.name} says: {message}"

    def move_to(self, position):
        """
        Moves the agent to a specified position.

        Parameters:
        position (int): The position to move the agent to.

        Returns:
        str: A message indicating the new position of the agent.
        """
        self.position = position
        return f"{self.name} moved to position {self.position}"

    def execute_trade(self, signal):
        """
        Executes a trade based on a given signal and logs the action.

        Parameters:
        signal (str): The trading signal to execute.

        Returns:
        str: A message indicating the result of the trade execution.
        """
        result = f"{self.name} executed {signal} trade"
        self.action_log.append(result)
        return result

    def get_status(self):
        """
        Returns the current status of the agent, including its name, position, and last action.

        Returns:
        dict: A dictionary containing the agent's name, position, and last action.
        """
        return {
            "name": self.name,
            "position": self.position,
            "last_action": self.action_log[-1] if self.action_log else "none"
        }
