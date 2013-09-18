from tkinter.constants import *
from settings import *
from safetkinter import *

import _thread
import collections
import itertools
import logging
import math
import os
import pickle
import random
import re
import string
import sys
import textwrap
import time
import tkinter.messagebox
import traceback

import me
import threadbox

################################################################################

class WabolTalk(Frame):

    TEXT = dict(width=25, height=4, wrap=WORD)
    GRID = dict(padx=5, pady=5)
    FILE = 'textarea.sav'
    ERR1 = 'ERROR: There is not enough public text to cover the private text.'
    ERR2 = 'ERROR: This duplicated message may not be longer than the primer.'

    @classmethod
    def main(cls):
        root = Tk()
        root.title('Wabol Talk 0.9.1 Beta')
        root.minsize(260, 350)
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.bind_all('<Control-Key-a>', cls.handle_control_a)
        root.bind_class('Text', '<Control-Key-/>', lambda event: 'break')
        frame = cls(root)
        frame.grid(sticky=NSEW)
        root.mainloop()

    @staticmethod
    def handle_control_a(event):
        widget = event.widget
        if isinstance(widget, Text):
            widget.tag_add(SEL, 1.0, END + '-1c')
            return 'break'
        if isinstance(widget, Entry):
            widget.selection_range(0, END)
            return 'break'

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.build_widgets()
        self.place_widgets()
        self.operation_lock = _thread.allocate_lock()
        self.options(True)
        self.message = Namespace(public=Parameter(''),
                                 private=Parameter(''),
                                 wabol=Parameter(''))
        self.load_file()

    def build_widgets(self):
        self.public_frame = Labelframe(self, text='Public Message:')
        self.public_text = ScrolledText(self.public_frame, **self.TEXT)
        self.private_frame = Labelframe(self, text='Private Message:')
        self.private_text = ScrolledText(self.private_frame, **self.TEXT)
        self.button_frame = Frame(self)
        self.encode_button = Button(self.button_frame, command=self.encode,
                                    text='\u02c5 Encode \u02c5')
        self.option_button = Button(self.button_frame, command=self.options,
                                    text='Options')
        self.decode_button = Button(self.button_frame, command=self.decode,
                                    text='\u02c4 Decode \u02c4')
        self.wabol_frame = Labelframe(self, text='Wabol Message:')
        self.wabol_text = ScrolledText(self.wabol_frame, **self.TEXT)
        self.sensitive_widgets = (self.public_text, self.private_text,
                                  self.encode_button, self.option_button,
                                  self.decode_button, self.wabol_text)

    def encode(self):
        if self.with_operation_lock(True):
            self.operate(self.do_encode)

    def decode(self):
        if self.with_operation_lock(True):
            self.operate(self.do_decode)

    def options(self, loading=False):
        if self.with_operation_lock(True):
            self.operate(self.do_options, loading)

    def operate(self, func, *args, **kwargs):
        start_thread(self.operation_thread, func, args, kwargs)

    @threadbox.MetaBox.thread
    def operation_thread(self, func, args, kwargs):
        try:
            func(*args, **kwargs)
        finally:
            self.with_operation_lock(False)

    def with_operation_lock(self, start):
        if start:
            acquired = self.operation_lock.acquire(False)
            if acquired:
                for widget in self.sensitive_widgets:
                    widget['state'] = DISABLED
            return acquired
        for widget in self.sensitive_widgets:
            widget['state'] = NORMAL
        self.operation_lock.release()

    @threadbox.MetaBox.thread
    def do_encode(self):
        public = self.public_data
        valid, message = self.validate_encode(
            public, self.encode_length(self.private_data), self.primer.data)
        if valid:
            private = me.encrypt_str(message, self.key, self.primer)[0]
            message = utility_encode(public, private)
        self.wabol_data = message

    @classmethod
    def validate_encode(cls, public, private, primer_data):
        extra = len(public) - len(private)
        if extra < 0:
            return False, cls.ERR1
        if extra > 0:
            choice = random.SystemRandom().choice
            private += ''.join(choice(string.printable) for _ in range(extra))
        node, primer_len = \
              large_duplicate(primer_data.decode() + private), len(primer_data)
        if node.length > primer_len:
            index = node.same[0].offset - primer_len
            return False, '{!s}\n\n{!r}'.format(
                cls.ERR2, private[max(index, 0):index+node.length])
        return True, private

    @threadbox.MetaBox.thread
    def do_decode(self):
        self.public_data, private = utility_decode(self.wabol_data)
        private = me.decrypt_str(private, self.key, self.primer)[0]
        self.private_data = self.decode_length(private)

    @threadbox.MetaBox.thread
    def do_options(self, loading):
        result = Options(self, 64, 'settings.sav', loading).result
        if result is not None:
            self.key, self.primer = result
            assert isinstance(self.key, Key) and \
                   isinstance(self.primer, Primer), 'Key/Primer of bad type!'
        elif loading:
            raise RuntimeError('Program was not able to load properly!')

    @staticmethod
    def encode_length(string, symbols=string.printable):
        length, (null, *base), prefix = len(string), symbols, ''
        while length:
            length, index = divmod(length, len(base))
            prefix = base[index] + prefix
        return prefix + null + string

    @staticmethod
    def decode_length(string, symbols=string.printable):
        length, (null, *base) = 0, symbols
        if null not in string:
            return ''
        prefix, string = string.split(null, 1)
        for char in prefix:
            length = length * len(base) + base.index(char)
        return string[:length]

    def place_widgets(self):
        self.public_frame.grid(sticky=NSEW, **self.GRID)
        self.public_frame.grid_rowconfigure(0, weight=1)
        self.public_frame.grid_columnconfigure(0, weight=1)
        self.public_text.grid(sticky=NSEW, **self.GRID)

        self.private_frame.grid(sticky=NSEW, **self.GRID)
        self.private_frame.grid_rowconfigure(0, weight=1)
        self.private_frame.grid_columnconfigure(0, weight=1)
        self.private_text.grid(sticky=NSEW, **self.GRID)

        self.button_frame.grid(sticky=EW, **self.GRID)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        self.encode_button.grid(row=0, column=0, sticky=EW, **self.GRID)
        self.option_button.grid(row=0, column=1, sticky=EW, **self.GRID)
        self.decode_button.grid(row=0, column=2, sticky=EW, **self.GRID)

        self.wabol_frame.grid(sticky=NSEW, **self.GRID)
        self.wabol_frame.grid_rowconfigure(0, weight=1)
        self.wabol_frame.grid_columnconfigure(0, weight=1)
        self.wabol_text.grid(sticky=NSEW, **self.GRID)

    def destroy(self):
        self.save_file()
        super().destroy()

    def load_file(self, path=None):
        time.sleep(0.001)   # Allow other threads to run.
        self.after_idle(self.idle_load, self.FILE if path is None else path)

    def idle_load(self, path):
        if self.with_operation_lock(True):
            self.operate(self.do_load, path)
        else:
            self.load_file(path)

    def do_load(self, path):
        try:
            self.message.load(path)
        except IOError:
            pass
        self.public_data, self.private_data, self.wabol_data = \
            self.message.public, self.message.private, self.message.wabol

    def save_file(self, path=None):
        if path is None:
            path = self.FILE
        self.message.public, self.message.private, self.message.wabol = \
            self.public_data, self.private_data, self.wabol_data
        self.message.save(path)

    ########################################################################

    def __get_public_text(self):
        return self.__get_text(self.public_text)

    def __set_public_text(self, value):
        self.__set_text(self.public_text, value)

    public_data = property(__get_public_text, __set_public_text)

    def __get_private_text(self):
        return self.__get_text(self.private_text)

    def __set_private_text(self, value):
        self.__set_text(self.private_text, value)

    private_data = property(__get_private_text, __set_private_text)

    def __get_wabol_text(self):
        return self.__get_text(self.wabol_text)

    def __set_wabol_text(self, value):
        self.__set_text(self.wabol_text, value)

    wabol_data = property(__get_wabol_text, __set_wabol_text)

    def __get_text(self, widget):
        return self.clean(widget.get(1.0, END + '-1c'))

    def __set_text(self, widget, value):
        state = widget['state']
        widget['state'] = NORMAL
        widget.delete(1.0, END)
        widget.insert(END, self.clean(value))
        widget['state'] = state

    @staticmethod
    def clean(text, allow=string.printable, hold='?'):
        return ''.join((char if char in allow else hold) for char in text)

################################################################################

def start_thread(function, *args, **kwargs):
    _thread.start_new_thread(log_errors, (function, args, kwargs))

def log_errors(function, args=(), kwargs={}):
    try:
        function(*args, **kwargs)
    except SystemExit:
        pass
    except:
        basename = os.path.basename(sys.argv[0])
        filename = os.path.splitext(basename)[0] + '.log'
        logging.basicConfig(filename=filename)
        logging.error(traceback.format_exc())

################################################################################

# See "match_and_compress.py" along with "match2.py" for original development.

def large_duplicate(array):
    for node in find_duplicates(array):
        pass
    return node

def find_duplicates(array):
    size, fork = len(array), {}
    seed = [Area(index, size - index) for index in range(size)]
    for area in seed:
        fork.setdefault(array[area.offset], []).append(area)
    root = Node(seed)
    path = collections.deque(root.make(same) for same in fork.values())
    yield root
    while path:
        node, fork = path.popleft(), {}
        for area in node.same:
            if area.length > node.length:
                fork.setdefault(
                    array[area.offset + node.length], []).append(area)
        for same in fork.values():
            if len(same) > 1:
                path.append(node.make(same))
        yield node

class Area:

    me.slots('offset, length')

    def __init__(self, offset, length):
        assert offset >= 0, 'Offset may not be negative!'
        assert length > 0, 'Length must be positive!'
        self.__offset, self.__length = offset, length

    @property
    def offset(self):
        return self.__offset

    @property
    def length(self):
        return self.__length

class Node:

    me.slots('same, length, parent, extent')

    def __init__(self, same, length=0, parent=None):
        self.__same, self.__length, self.__parent, self.__extent = \
            same, length, parent, []

    def make(self, same):
        return Node(same, self.__length + 1)
        # Doubly linked tree is made below.
        self.__extent.append(Node(same, self.__length + 1, self))
        return self.__extent[-1]

    @property
    def same(self):
        return self.__same

    @property
    def length(self):
        return self.__length

    @property
    def parent(self):
        return self.__parent

    @property
    def extent(self):
        return self.__extent

################################################################################

class Key(me.Key):

    me.slots()

    @classmethod
    def new_deterministic(cls, bytes_used, chain_size):
        selection, blocks, chaos = list(set(bytes_used)), [], random.Random()
        chaos.seed(chain_size.to_bytes(math.ceil(
            chain_size.bit_length() / 8), 'big') + bytes(range(256)))
        for _ in range(chain_size):
            chaos.shuffle(selection)
            blocks.append(bytes(selection))
        return cls(tuple(blocks))

    @classmethod
    def new_client_random(cls, bytes_used, chain_size, chaos):
        selection, blocks = list(set(bytes_used)), []
        for _ in range(chain_size):
            chaos.shuffle(selection)
            blocks.append(bytes(selection))
        return cls(tuple(blocks))

################################################################################

class Primer(me.Primer):

    me.slots()

    @classmethod
    def new_deterministic(cls, key):
        base, chain_size, chaos = key.base, key.dimensions, random.Random()
        chaos.seed(chain_size.to_bytes(math.ceil(
            chain_size.bit_length() / 8), 'big') + bytes(range(256)))
        return cls(bytes(chaos.choice(base) for _ in range(chain_size - 1)))

    @classmethod
    def new_client_random(cls, key, chaos):
        base = key.base
        return cls(bytes(chaos.choice(base) for _ in range(key.dimensions - 1)))

################################################################################

KEY = Key((b'T#cCpD*Ola35gbX-[(sB}\'G+&$@7Kd~Sx_Z>y%0QU"8Pv?YWM;fNhVqLj)|4{,tn.!wJAr^1R/HFiumz6:`eo2<I9\\]kE=',
           b'UyA_bt~"r?c^h7BH5vK|WSZ%g.QIMs/ET12Xl=u>+},PwL-o6Y{aGq4]9RJ&j[mV\\;fxNizedOD\'!n$@F0p8(:`C#)3k<*',
           b'n`yo;b/TtVN0f%Q.k_Zg#KMDx9jAqa1<\\+e!}O)l?48>mYB:7^w|2G*$CW-SdIi{@s=hHX,z~c6r"E[UpR5PJu]&\'v(3FL'))

PRIMER = Primer(b'P9')

MAIN_SYMBOLS = string.printable

ICON_SYMBOLS = tuple(''.join(word) for word in itertools.permutations('wabol'))

################################################################################

def utility_encode(public, private):
    text = ''.join(interlace(private, public))
    code = me.encrypt_str(text, KEY, PRIMER)[0]
    values = encode_to_values(code)
    return textwrap.fill(' '.join(encode_to_repr(values)))

def utility_decode(wabol):
    values = decode_to_values(wabol)
    code = ''.join(decode_to_repr(values))
    text = me.decrypt_str(code, KEY, PRIMER)[0]
    return text[1::2], text[0::2]

def interlace(*args):
    data = []
    for items in zip(*args):
        data.extend(items)
    return data

################################################################################

def encode_to_values(string, max_bits=1024):
    source, destination = SymbolTable(MAIN_SYMBOLS), SymbolTable(ICON_SYMBOLS)
    tokenize = Tokenizer(source)
    *target, delimiter = destination
    base1, value, base2 = len(source) + 1, 0, len(target)
    for item in tokenize(string):
        value = value * base1 + source[item] + 1
        if value.bit_length() >= max_bits:
            yield value, target, delimiter, base2
            value = 0
    if value:
        yield value, target, delimiter, base2

def encode_to_repr(iterable, first=True):
    for value, target, delimiter, base in iterable:
        if first:
            first = False
        else:
            yield delimiter
        stack = collections.deque()
        while value:
            value, index = divmod(value, base)
            stack.appendleft(target[index])
        yield ' '.join(stack)

################################################################################

def decode_to_values(string):
    source, destination = SymbolTable(ICON_SYMBOLS), SymbolTable(MAIN_SYMBOLS)
    tokenize = Tokenizer(source)
    *origin, delimiter = source
    origin, base1, base2 = \
        SymbolTable(origin), len(origin), len(destination) + 1
    for block in string.split(delimiter):
        value = 0
        for item in tokenize(block):
            value = value * base1 + origin[item]
        yield value, destination, base2

def decode_to_repr(iterable):
    for value, target, base in iterable:
        stack = collections.deque()
        while value:
            value, index = divmod(value, base)
            if not index:
                raise ValueError('There was an error in decoding!')
            stack.appendleft(target[index - 1])
        yield ''.join(stack)

################################################################################

class SymbolTable:

    me.slots('symbols')

    def __init__(self, iterable):
        array = sorted(iterable, key=len, reverse=True)
        unique, total = set(array), len(array)
        if len(unique) != total:
            raise ValueError('Symbol table must have unique elements!')
        if total < 3:
            raise ValueError('There must be more than two symbols!')
        self.__symbols = tuple(array)

    def __len__(self):
        return len(self.__symbols)

    def __iter__(self):
        return iter(self.__symbols)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self.__symbols[key]
        return self.__symbols.index(key)

################################################################################

class Tokenizer:

    me.slots('engine')

    def __init__(self, table, flags=0):
        self.__engine = re.compile('|'.join(map(re.escape, table)), flags)

    def __call__(self, string):
        for match in self.__engine.finditer(string):
            yield match.group()

################################################################################

class Dialog(Toplevel): # Copies tkinter.simpledialog.Dialog

    def __init__(self, parent, title=None):
        super().__init__(parent)
        self.withdraw()
        if parent.winfo_viewable():
            self.transient(parent)
        if title:
            self.title(title)
        self.parent = parent
        self.result = None
        body = Frame(self)
        self.initial_focus = self.body(body)
        body.grid(sticky=NSEW, padx=5, pady=5)
        self.buttonbox()
        if not self.initial_focus:
            self.initial_focus = self
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        if self.parent is not None:
            self.geometry('+{}+{}'.format(parent.winfo_rootx() + 50,
                                          parent.winfo_rooty() + 50))
        self.deiconify()
        self.initial_focus.focus_set()
        try:
            self.wait_visibility()
        except tkinter.TclError:
            pass
        else:
            self.grab_set()
            self.wait_window(self)

    def destroy(self):
        self.initial_focus = None
        super().destroy()

    def body(self, master):
        pass

    def buttonbox(self):
        box = Frame(self)
        w = Button(box, text='OK', width=10, command=self.ok, default=ACTIVE)
        w.grid(row=0, column=0, padx=5, pady=5)
        w = Button(box, text='Cancel', width=10, command=self.cancel)
        w.grid(row=0, column=1, padx=5, pady=5)
        self.bind('<Return>', self.ok)
        self.bind('<Escape>', self.cancel)
        box.grid()

    def ok(self, event=None):
        if not self.validate():
            self.initial_focus.focus_set()
            return
        self.withdraw()
        self.update_idletasks()
        try:
            self.apply()
        finally:
            self.cancel()

    def cancel(self, event=None):
        if self.parent is not None:
            self.parent.focus_set()
        self.destroy()

    def validate(self):
        return True

    def apply(self):
        pass

################################################################################

class Options(Dialog):

    GRID = WabolTalk.GRID
    PRESET, SINGLE, DOUBLE = 'preset single double'.split()
    KEY, PRIMER, KEYPRIMER = 'key primer keyprimer'.split()
    USED = {PRESET, SINGLE, DOUBLE}
    KEYS = {KEY, PRIMER, KEYPRIMER}

    def __init__(self, parent, min_bits, file_path, loading):
        if not isinstance(min_bits, int):
            raise TypeError('type(min_bits) = ' + repr(type(min_bits)))
        if min_bits < 1:
            raise ValueError('min_bits = ' + repr(min_bits))
        self.min_bits = min_bits
        self.file_path = file_path
        self.loading = loading
        super().__init__(parent, 'Options')

    def body(self, master):
        self.resizable(True, False)
        self.minsize(215, 1)
        self.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        self.build_widgets(master)
        self.make_data_table()
        self.place_widgets()
        self.setup_widgets()
        self.load_file()
        self.handle_hint_used()
        if self.loading:
            self.after_idle(self.ok)

    def build_widgets(self, master):
        self.parameter_frame = Labelframe(master, text='Parameters:')
        self.primer_var = StringVar(master)
        self.primer_label = Label(self.parameter_frame, text='Primer Length:')
        self.primer_spinbox = Spinbox(self.parameter_frame, from_=16, to=256,
                                      textvariable=self.primer_var)
        self.hint_a_var = StringVar(master)
        self.hint_a_label = Label(self.parameter_frame, text='Hint A:')
        self.hint_a_entry = Entry(self.parameter_frame,
                                  textvariable=self.hint_a_var)
        self.hint_b_var = StringVar(master)
        self.hint_b_label = Label(self.parameter_frame, text='Hint B:')
        self.hint_b_entry = Entry(self.parameter_frame,
                                  textvariable=self.hint_b_var)
        
        self.hint_used_frame = Labelframe(master, text='K/P Creation:')
        self.hint_used_space = Frame(self.hint_used_frame)
        self.hint_used_var = StringVar(master)
        self.preset_button = Radiobutton(self.hint_used_space,
                                         text='Use Preset Codes',
                                         command=self.handle_hint_used,
                                         value=self.PRESET,
                                         variable=self.hint_used_var)
        self.single_button = Radiobutton(self.hint_used_space,
                                         text='Allow Single Hint',
                                         command=self.handle_hint_used,
                                         value=self.SINGLE,
                                         variable=self.hint_used_var)
        self.double_button = Radiobutton(self.hint_used_space,
                                         text='Allow Double Hints',
                                         command=self.handle_hint_used,
                                         value=self.DOUBLE,
                                         variable=self.hint_used_var)
        
        self.hint_pick_frame = Labelframe(master, text='Application:')
        self.hint_pick_space = Frame(self.hint_pick_frame)
        self.hint_pick_var = StringVar(master)
        self.use_key_button = Radiobutton(self.hint_pick_space,
                                          text='Change Key',
                                          command=self.handle_hint_pick,
                                          value=self.KEY,
                                          variable=self.hint_pick_var)
        self.use_primer_button = Radiobutton(self.hint_pick_space,
                                             text='Change Primer',
                                             command=self.handle_hint_pick,
                                             value=self.PRIMER,
                                             variable=self.hint_pick_var)
        self.use_keyprimer_button = Radiobutton(self.hint_pick_space,
                                                text='Alter Key/Primer',
                                                command=self.handle_hint_pick,
                                                value=self.KEYPRIMER,
                                                variable=self.hint_pick_var)
        
        self.use_order_frame = Labelframe(master, text='Priority:')
        self.use_order_space = Frame(self.use_order_frame)
        self.use_order_var = StringVar(master)
        self.pri_key_button = Radiobutton(self.use_order_space,
                                          text='Create Key First',
                                          command=self.handle_use_order,
                                          value=self.KEY,
                                          variable=self.use_order_var)
        self.pri_primer_button = Radiobutton(self.use_order_space,
                                             text='Create Primer First',
                                            command=self.handle_use_order,
                                             value=self.PRIMER,
                                             variable=self.use_order_var)
        self.pri_keyprimer_button = Radiobutton(self.use_order_space,
                                                text='Assign Same Importance',
                                                command=self.handle_use_order,
                                                value=self.KEYPRIMER,
                                                variable=self.use_order_var)

    def make_data_table(self):
        used, keys = self.USED, self.KEYS
        self.options = Namespace(
            primer=Parameter(16, lambda value: value in range(16, 257)),
            hint_a=Parameter(''),
            hint_b=Parameter(''),
            key=Namespace(
                hint_used=Parameter(self.PRESET, lambda value: value in used),
                hint_pick=Parameter(self.KEY, lambda value: value in keys),
                use_order=Parameter(self.KEY, lambda value: value in keys)))
        self.data_table = ((self.primer_var, 'primer'),
                           (self.hint_a_var, 'hint_a'),
                           (self.hint_b_var, 'hint_b'),
                           (self.hint_used_var, 'key.hint_used'),
                           (self.hint_pick_var, 'key.hint_pick'),
                           (self.use_order_var, 'key.use_order'))
        return used, keys

    def handle_hint_used(self):
        hint_used = self.hint_used_var.get()
        if hint_used == self.PRESET:
            self.hint_a_label.grid_remove()
            self.hint_a_entry.grid_remove()
            self.hint_b_label.grid_remove()
            self.hint_b_entry.grid_remove()
            self.hint_pick_frame.grid_remove()
            self.use_order_frame.grid_remove()
        elif hint_used == self.SINGLE:
            self.hint_a_label.grid()
            self.hint_a_entry.grid()
            self.hint_b_label.grid_remove()
            self.hint_b_entry.grid_remove()
            self.handle_hint_pick()
        elif hint_used == self.DOUBLE:
            self.hint_a_label.grid()
            self.hint_a_entry.grid()
            self.hint_b_label.grid()
            self.hint_b_entry.grid()
            self.hint_pick_frame.grid_remove()
            self.use_order_frame.grid_remove()
        else:
            raise ValueError('hint_used = ' + repr(hint_used))

    def handle_hint_pick(self):
        self.hint_pick_frame.grid()
        hint_pick = self.hint_pick_var.get()
        if hint_pick == self.KEY:
            self.use_order_frame.grid_remove()
        elif hint_pick == self.PRIMER:
            self.use_order_frame.grid_remove()
        elif hint_pick == self.KEYPRIMER:
            self.handle_use_order()
        else:
            raise ValueError('hint_pick = ' + repr(hint_pick))

    def handle_use_order(self):
        self.use_order_frame.grid()
        use_order = self.use_order_var.get()
        if use_order == self.KEY:
            pass
        elif use_order == self.PRIMER:
            pass
        elif use_order == self.KEYPRIMER:
            pass
        else:
            raise ValueError('use_order = ' + repr(use_order))

    def place_widgets(self):
        self.parameter_frame.grid(sticky=EW, **self.GRID)
        self.parameter_frame.grid_columnconfigure(1, weight=1)
        self.primer_label.grid(row=0, column=0, sticky=W, **self.GRID)
        self.primer_spinbox.grid(row=0, column=1, sticky=EW, **self.GRID)
        self.hint_a_label.grid(row=1, column=0, sticky=W, **self.GRID)
        self.hint_a_entry.grid(row=1, column=1, sticky=EW, **self.GRID)
        self.hint_b_label.grid(row=2, column=0, sticky=W, **self.GRID)
        self.hint_b_entry.grid(row=2, column=1, sticky=EW, **self.GRID)
        
        self.hint_used_frame.grid(sticky=EW, **self.GRID)
        self.hint_used_frame.grid_columnconfigure(0, weight=1)
        self.hint_used_space.grid()
        self.preset_button.grid(sticky=W, **self.GRID)
        self.single_button.grid(sticky=W, **self.GRID)
        self.double_button.grid(sticky=W, **self.GRID)
        
        self.hint_pick_frame.grid(sticky=EW, **self.GRID)
        self.hint_pick_frame.grid_columnconfigure(0, weight=1)
        self.hint_pick_space.grid()
        self.use_key_button.grid(sticky=W, **self.GRID)
        self.use_primer_button.grid(sticky=W, **self.GRID)
        self.use_keyprimer_button.grid(sticky=W, **self.GRID)
        
        self.use_order_frame.grid(sticky=EW, **self.GRID)
        self.use_order_frame.grid_columnconfigure(0, weight=1)
        self.use_order_space.grid()
        self.pri_key_button.grid(sticky=W, **self.GRID)
        self.pri_primer_button.grid(sticky=W, **self.GRID)
        self.pri_keyprimer_button.grid(sticky=W, **self.GRID)

    def setup_widgets(self):
        self.primer_spinbox.bind('<Key>', lambda event: 'break')
        self.hint_a_entry.bind('<Control-Key-/>', lambda event: 'break')
        self.hint_b_entry.bind('<Control-Key-/>', lambda event: 'break')

    def validate(self):
        hint_used = self.hint_used_var.get()
        if hint_used in {self.SINGLE, self.DOUBLE}:
            if self.error('a') or hint_used == self.DOUBLE and self.error('b'):
                return False
        return True

    def error(self, name):
        try:
            hint = getattr(self, 'hint_{}_var'.format(name)).get()
        except UnicodeDecodeError:
            quality = -1
        else:
            quality = self.estimate_quality(hint)
        if quality < 0:
            tkinter.messagebox.showerror(
                'Illegal Character',
                'Hint {} contains unacceptable text.'.format(name.upper()),
                master=self)
            return True
        elif quality < self.min_bits:
            tkinter.messagebox.showerror(
                'Insecure Password',
                'Hint {} must have greater security.'.format(name.upper()),
                master=self)
            return True
        return False

    @staticmethod
    def estimate_quality(password):
        if not password:
            return 0
        types = ([False, frozenset(string.whitespace)],
                 [False, frozenset(string.ascii_lowercase)],
                 [False, frozenset(string.ascii_uppercase)],
                 [False, frozenset(string.digits)],
                 [False, frozenset(string.punctuation)])
        for character in password:
            for index, (flag, group) in enumerate(types):
                if character in group:
                    types[index][0] = True
                    break
            else:
                return -1
        space = sum(len(group) for flag, group in types if flag)
        return math.ceil(math.log2(space) * len(password))

    def apply(self):
        self.save_file()
        hint_used = self.hint_used_var.get()
        if hint_used == self.PRESET:
            self.result = self.get_preset()
        elif hint_used == self.SINGLE:
            self.result = self.get_single()
        elif hint_used == self.DOUBLE:
            self.result = self.get_double()
        else:
             raise ValueError('hint_used = ' + repr(hint_used))

    def get_preset(self):
        key = Key.new_deterministic(string.printable.encode(),
                                    int(self.primer_var.get()) + 1)
        primer = Primer.new_deterministic(key)
        return key, primer

    def get_single(self):
        chaos = random.Random()
        chaos.seed(self.hint_a_var.get())
        hint_pick = self.hint_pick_var.get()
        if hint_pick == self.KEY:
            key = Key.new_client_random(string.printable.encode(),
                                        int(self.primer_var.get()) + 1,
                                        chaos)
            primer = Primer.new_deterministic(key)
            return key, primer
        elif hint_pick == self.PRIMER:
            key = Key.new_deterministic(string.printable.encode(),
                                        int(self.primer_var.get()) + 1)
            primer = Primer.new_client_random(key, chaos)
            return key, primer
        elif hint_pick == self.KEYPRIMER:
            return self.get_single_keyprimer(chaos)
        else:
            raise ValueError('hint_pick = ' + repr(hint_pick))

    def get_single_keyprimer(self, chaos):
        use_order = self.use_order_var.get()
        if use_order == self.KEY:
            key = Key.new_client_random(string.printable.encode(),
                                        int(self.primer_var.get()) + 1,
                                        chaos)
            primer = Primer.new_client_random(key, chaos)
            return key, primer
        elif use_order == self.PRIMER:
            key = Key.new_client_random(string.printable.encode(),
                                        int(self.primer_var.get()) + 1,
                                        chaos)
            chaos.seed(self.hint_a_var.get())
            primer = Primer.new_client_random(key, chaos)
            key = Key.new_client_random(string.printable.encode(),
                                        int(self.primer_var.get()) + 1,
                                        chaos)
            return key, primer
        elif use_order == self.KEYPRIMER:
            key = Key.new_client_random(string.printable.encode(),
                                        int(self.primer_var.get()) + 1,
                                        chaos)
            chaos.seed(self.hint_a_var.get())
            primer = Primer.new_client_random(key, chaos)
            return key, primer
        else:
            raise ValueError('use_order = ' + repr(use_order))

    def get_double(self):
        chaos = random.Random()
        chaos.seed(self.hint_a_var.get())
        key = Key.new_client_random(string.printable.encode(),
                                    int(self.primer_var.get()) + 1,
                                    chaos)
        chaos.seed(self.hint_b_var.get())
        primer = Primer.new_client_random(key, chaos)
        return key, primer

    def load_file(self):
        try:
            self.options.load(self.file_path)
        except IOError:
            pass
        for var, name in self.data_table:
            var.set(self.options[name])

    def save_file(self):
        for var, name in self.data_table:
            self.options[name] = type(self.options[name])(var.get())
        self.options.save(self.file_path)

################################################################################

# Used to create Tkinter root in another thread.

def get_root():
    data = [_thread.allocate_lock(),
            lambda: (data.append(Tk()), data[0].release(), data[2].mainloop())]
    with data[0]:
        _thread.start_new_thread(data[1], ())
        data[0].acquire()
    return data[2]

################################################################################

if __name__ == '__main__':
    log_errors(WabolTalk.main)
