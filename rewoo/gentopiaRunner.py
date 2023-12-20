import dotenv
from gentopia import chat
from gentopia.assembler.agent_assembler import AgentAssembler

dotenv.load_dotenv(".env")  # load environmental keys
agent = AgentAssembler(file='agent.yaml').get_agent()

chat(agent) 

# oh = Your_OutputHandler()
# assert isinstance(oh, BaseOutput)
# # Response content streams into your output handler.
# agent.stream("Your instruction or message.", output_handler=oh)