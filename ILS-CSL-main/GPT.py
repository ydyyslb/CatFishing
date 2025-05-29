
import json
import openai
from retry import retry
from rich import print as rprint


class GPT:
    def __init__(self, api_key="sk-lILX67qwKgjQxyht6a36F8E6E5Bf4bD29295765c148886Cf", model="gpt-4", temperature=0.7):
        self.model = model
        self.temperature = temperature
        openai.api_key = api_key
        openai.api_base = "https://api.kwwai.top/v1"  # 修改为你的代理API

    @retry(tries=5, delay=2, backoff=2, jitter=(1, 3), logger=None)
    def chatgpt_QA(self, question, outfile_path=None, quiet=False):
        """
        使用OpenAI的API与GPT模型进行问答交互。
        """
        result = {"model": self.model, "input": question, "output": None}  # 初始化输出为None
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=self.temperature,
                max_tokens=5000  # 可根据需要调整
            )
            result["output"] = response.choices[0].message["content"].strip()  # 更新输出

            if not quiet:
                rprint(f"✅ [green]Successfully[/green] queried the [bold yellow]{self.model}[/bold yellow].")

            if outfile_path:
                with open(outfile_path, "w", encoding="utf-8") as outfile:
                    json.dump(response, outfile, indent=4, ensure_ascii=False)

        except Exception as e:
            rprint(f"❌ [red]Failed[/red] to query the [bold yellow]{self.model}[/bold yellow], retrying...")
            # 不要在这里抛出异常，而是可以记录日志或者采取其他错误处理措施
            # raise e  # 这行代码应该被注释掉或删除

        return result


if __name__ == '__main__':
    prompts = "You are an expert on childhood diseases."
    api_key = "sk-lILX67qwKgjQxyht6a36F8E6E5Bf4bD29295765c148886Cf"
    chatgpt = GPT(api_key, model="gpt-4", temperature=0.7)

    output = chatgpt.chatgpt_QA(prompts)
    print(output["output"])
