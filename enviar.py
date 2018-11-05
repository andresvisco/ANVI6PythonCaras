import asyncio

async def enviar(datos):
    print(datos)
    

def main():
    loop=asyncio.get_event_loop()
    loop.run_until_complete(enviar("hola"))
    loop.close()

main()
