# coding: utf-8
#  Designed for Bluetooth LE on Windows 8/10
#  Author:  CaptainSmiley & Warren (CyKIT)
try:
    import sys
    import binascii
    import os
    import struct
    import numpy as np
    from time import sleep, time
    from ctypes import *
    from base64 import b64decode
    from ctypes.wintypes import HANDLE, ULONG, DWORD, USHORT
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad
except (ImportError, IOError) as e:
    print("ERROR:  %s" % e)
    sys.exit(-1)

DEVICE_UUID = "{81072f40-9f3d-11e3-a9dc-0002a5d5c51b}"
DATA_UUID = "{81072f41-9f3d-11e3-a9dc-0002a5d5c51b}"
MEMS_UUID = "{81072f42-9f3d-11e3-a9dc-0002a5d5c51b}"

BLOCK_SIZE = 16

cipher = ""
start_recording = False

class BTH_LE_GATT_CHARACTERISTIC_VALUE(Structure):
    _fields_ = [
        ("DataSize", c_ulong),
        ("Data", c_ubyte * 20)]         #  this is a hack - 20-bytes should be dynamic, not preallocated
    
_CB_FUNC_ = CFUNCTYPE(None, BTH_LE_GATT_CHARACTERISTIC_VALUE)
_ERR_FUNC_ = CFUNCTYPE(None, c_wchar_p)

def PrintLibError(err):
    print(str(err));

def StreamToQueue(s = 10):
    global cipher, EEGqueue
    
    eegDll = cdll.LoadLibrary(os.getcwd() + "\\EEGBtleLib\\Win32\\Release\\Win32EEGBtleLib.dll")

    #  these are the UUIDs of the data streams we want to decrypt
    #  uuid_list = [unicode(DATA_UUID), unicode(MEMS_UUID)]
    uuid_list = [unicode(DATA_UUID)]
    uuid_clist = (c_wchar_p * len(uuid_list))()
    uuid_clist[:] = uuid_list

    #  set the error callback function to handle lib file errors
    err_func = _ERR_FUNC_(PrintLibError)
    eegDll.set_error_func(err_func)

    #  initialize the dll engine; pass it the UUID of the device we want to connect too
    hDev = eegDll.btle_init(unicode(DEVICE_UUID))

    #  set the callback function to handle the updates
    cb_func = _CB_FUNC_(DataCallback_fill_queue)
    eegDll.set_callback_func(cb_func)

    #  we get the BTLE device ID from the connected device
    name = eegDll.get_bluetooth_id()

    print("Getting connection info...")
    while_timeout = time() + 10
    while c_wchar_p(name).value is None and time() < while_timeout:
        name = eegDll.get_bluetooth_id()
        sleep(1)

    name = c_wchar_p(name).value
    name_id = name[name.find("(")+1:name.find(")")]
    print("Connected headset:  %s" % name)

    #  generate the AES decryption key from the device ID
    sn = bytearray(name_id.decode("hex"))
    k = [sn[-4],sn[-3],sn[-3],sn[-2],sn[-2],sn[-2],sn[-3],sn[-1],sn[-4],sn[-1],sn[-3],sn[-3],sn[-1],sn[-1],sn[-3],sn[-4]]
    AES_key = bytes(bytearray(k)) 

    #print("AES key: " + binascii.hexlify(AES_key) + "\n")

    #  initalize the decryption key function
    cipher = AES.new(AES_key, AES.MODE_ECB)
        
    #  start the data collection; pass it the list of UUID characteristics we are
    #  interested in
    eegDll.run_data_collection(hDev, uuid_clist)
    
    sleep(10)

    #  disconnect from the device
    eegDll.btle_disconnect(hDev)
    print("Done")
    return 0


def DataCallback_fill_queue(EventOutParameter):
    global start_recording, EEGqueue, Empty, Full
    data = bytearray(EventOutParameter.Data)

    #  we want to make sure that our decryption starts with the first of the two btle packets
    if start_recording is False:
        if data[1] == 0x01:
            start_recording = True

    #  decrypt the data stream
    if start_recording is True:
        #print("Ciphertext:\n" + binascii.hexlify(data))
        ciphertext = data[2:18]
        #print("Adjusted:\n" + binascii.hexlify(ciphertext))
        data_out = cipher.decrypt(ciphertext)
        #print("decrypted:\n" + binascii.hexlify(data_out) + "\n")
        eeg_chunk = np.zeros((7, 2))
        count = 1
        for i in range(2, 16, 2):        
            value_1 = int(binascii.hexlify(data_out[i]), 16)
            value_2 = int(binascii.hexlify(data_out[i+1]), 16)
            edk_value = np.around(((value_1 * .128205128205129) + 4201.02564096001) + ((value_2 - 128) * 32.82051289), 8)

            if data[1] == 0x01:
                eeg_chunk[:-1] = eeg_chunk[1:]
                eeg_chunk[-1] = count, edk_value
            if data[1] == 0x02:
                eeg_chunk[:-1] = eeg_chunk[1:]
                eeg_chunk[-1] = count + 8, edk_value
            count += 1
        
        try:
            EEGqueue.put(eeg_chunk, block = False)
        except Full:
            print("Filler exiting, timeout/full")
            sleep(0.001)
        except Empty:
            print("Empty, giving it a sec...")
            sleep(0.001)
        except Exception as e:
            print("Error:", e)