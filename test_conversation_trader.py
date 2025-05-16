import unittest
from conversation_trader import ConversationalTradeEngine
from avatar_agent import TradingAvatarAgent

class TestConversationalTradeEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ConversationalTradeEngine()

    def test_interpret_user_input_and_trigger_actions(self):
        response, action = self.engine.converse("Should I buy?")
        self.assertEqual(action, "suggest_buy")
        self.assertIn("buy might be valid", response)

        response, action = self.engine.converse("Sell now")
        self.assertEqual(action, "execute_sell")
        self.assertIn("Preparing to execute a sell order", response)

        response, action = self.engine.converse("What's your view?")
        self.assertEqual(action, "provide_outlook")
        self.assertIn("market looks uncertain", response)

    def test_agent_response_to_actions(self):
        self.engine.converse("Should I buy?")
        self.assertIn("OmniBot executed buy trade", self.engine.agent.action_log)

        self.engine.converse("Sell now")
        self.assertIn("OmniBot executed sell trade", self.engine.agent.action_log)

    def test_comprehensive_integration_flow(self):
        self.engine.converse("Should I buy?")
        self.engine.converse("Sell now")
        self.engine.converse("What's your view?")
        self.assertEqual(len(self.engine.dialogue_log), 3)
        self.assertEqual(self.engine.agent.position, 0)
        self.assertEqual(self.engine.agent.get_status()["last_action"], "OmniBot executed sell trade")

    def test_agent_state_maintenance(self):
        self.engine.converse("Should I buy?")
        self.engine.converse("Sell now")
        self.assertEqual(self.engine.agent.position, 0)
        self.assertEqual(self.engine.agent.get_status()["last_action"], "OmniBot executed sell trade")

    def test_action_log_accuracy(self):
        self.engine.converse("Should I buy?")
        self.engine.converse("Sell now")
        self.assertEqual(self.engine.agent.action_log, ["OmniBot executed buy trade", "OmniBot executed sell trade"])

if __name__ == '__main__':
    unittest.main()
